import torch
from detectron2.utils.env import _import_file as import_file

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from detectron2.utils import comm



from . import datasets as D
from . import samplers
from .build import build_dataset, make_batch_data_sampler
from fcos.core.data.collate_batch import BatchCollator, BBoxAugCollator
from .example_selector import ExampleSelector
from .rng_handler import RNGHandler
from .samplers.fs_sampler import FilteringSampler, SupportSampler
from .task_sampling import TaskSampler
from .transforms import build_transforms
from ..utils import HiddenPrints
import math


class DataHandler():
    def __init__(self,
                 cfg,
                 base_classes=True,
                 start_iter=0,
                 data_source='train',
                 eval_all=False,
                 is_train=False,
                 is_finetune=False, 
                 seed=None):
        """
        Object to manage datasets and dataloader. It can be used in a fewshot setting
        yielding loader for query and support sets or in a regular manner (only one loader).
        
        When eval_all is true sample batch of tasks to eval performances on all classes.
        - cfg: Config object for the whole network
        - base_classes: controls wether data should contains annotations either from base or novel classes
        - start_iter: start at iteration start_iter when restarting training
        - data_source: either 'train', 'val' or 'test' controls from which dataset split data is loaded
        - eval_all: special mode that samples test episodes and respective loader for evaluating on all classes (base and novel)
        - is_train: specifies whether network is training or not to select parameters accordingly
        - is_finetune: specifies whether network is finetuning or not. This changes how examples are selected in dataset
        """
        self.cfg = cfg
        self.base_classes = base_classes
        self.start_iter = start_iter
        self.data_source = data_source
        self.eval_all = eval_all
        self.is_train = is_train
        self.is_finetune = is_finetune
        self.categories = None
        self.selected_base_examples = {}
        self.selected_novel_examples = {}

        # To properly manage sampling for both base and novel classes two
        # rng handler are used. rng_handler_fixed's seed is fixed before the sampling of
        # the dataloader to fix it.
        self.rng_handler_fixed = RNGHandler(cfg)
        self.rng_handler_free = RNGHandler(cfg)
        if seed is not None:
            self.rng_handler_fixed.update_seeds(seed)

        self.example_selector = ExampleSelector(cfg, is_train, is_finetune, self.rng_handler_free)

        with HiddenPrints():
            self.datasets, self.support_datasets = self.build_datasets(cfg)

        self.task_sampler = TaskSampler(cfg,
                                        self.get_class_indices(),
                                        rng=self.rng_handler_fixed.rn_rng,
                                        eval=eval_all)

    def build_datasets(self, cfg):
        """
        Select right parameters and dataset class and create datasets
        both for query and support (when FS is enabled).
        """
        dataset_catalog = import_file("fcos.core.config.paths_catalog", self.cfg.PATHS_CATALOG, True).DatasetCatalog
        transforms = self.get_transforms(cfg)
        dataset_list = getattr(self.cfg.DATASETS, self.data_source.upper())
        mode = 'finetune' if self.is_finetune else 'train'

        datasets = {
            'query': build_dataset(dataset_list, transforms, dataset_catalog, cfg=self.cfg, is_train=self.is_train, mode=mode),
            'support': None
        }

        if cfg.FEWSHOT.ENABLED:
            if not self.is_train and self.cfg.FINETUNE.EXAMPLES == 'deterministic':
                dataset_list = getattr(self.cfg.DATASETS, 'TRAIN')
            datasets['support'] = build_dataset(dataset_list, transforms, dataset_catalog, cfg=self.cfg, is_train=self.is_train, mode='support')
        
        return datasets['query'], datasets['support']

    def get_transforms(self, cfg):
        # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
        if self.is_train and self.cfg.TEST.BBOX_AUG.ENABLED:
            return None
        return build_transforms(cfg, self.is_train)

    def get_class_indices(self):
        return list(range(1, len(self.datasets[0].coco.cats) + 1))
                
    def get_dataloader(self, seed=None):
        """
        Return either 1 dataloader for the whole training, with MAX_ITER 
        iteration. This is done when FSL is disabled.

        Or 2 dataloaders, 1 for query set and 1 for support set.

        """
        if seed is not None:
            self.rng_handler_fixed.update_seeds(seed)

        if self.cfg.FEWSHOT.ENABLED:
            return self.handle_fewshot_dataloader()
        else:
            return self.make_data_loader_filtered(self.get_class_indices(), self.datasets, is_fewshot=False)
        
    def handle_fewshot_dataloader(self):
        train_classes, self.test_classes = self.task_sampler.sample_train_val_tasks(self.cfg.FEWSHOT.N_WAYS_TRAIN, self.cfg.FEWSHOT.N_WAYS_TEST, verbose=False)
        if self.eval_all:
            # When eval_all is true, query and support loaders are created
            # for each batch of classes output from the task_sampler.
            # These are divided into train and test classes.
            loaders = {'train': [], 'test': []}
            
            for classes, kind in [(train_classes, 'train'), (self.test_classes, 'test')]:
                for cls in classes:
                    loaders[kind].append(self.get_two_loaders(cls, self.datasets, self.support_datasets))
            return loaders

        elif self.is_finetune and self.cfg.FINETUNE.MIXED:
            classes = self.task_sampler.finetuning_classes_selection(train_classes, self.test_classes, self.rng_handler_free)
        else:
            classes = train_classes if self.base_classes else self.test_classes

        return self.get_two_loaders(classes, self.datasets, self.support_datasets)
    
    def get_two_loaders(self, classes, datasets, support_datasets=None):
        """
        Arguments:
            classes: list of classes
            datasets: list of datasets list first element is query dataset, second is support

        Return two dataloaders: one for query set and one for support and the set of classes. 
        """
        # Assuming make_data_loader_filtered is where loaders are instantiated
        query_loader = self.make_data_loader_filtered(torch.Tensor(classes), datasets)
        support_loader = self.make_data_loader_filtered(torch.Tensor(classes), support_datasets, is_support=True)

        return query_loader, support_loader, classes

    def make_data_loader_filtered(self, selected_classes, datasets, is_support=False, is_fewshot=True, sampler=None):
        """
        Select parameters for the creation of the dataloader

        Arguments:
            selected_classes: list of classes that should be annotated in the loader
            datasets: list of dataset in which load the data
            is_support: controls whether it a support or query loader
            is_fewshot: controls whether loader should be for fewshot use or not
        
        Returns:
        """
        sampler_options = self.configure_sampler_options(is_fewshot)
        # DistributedSampler(dataset, shuffle=sampler_options['shuffle'])

        data_loaders = []
        for dataset in datasets: # will work only for cocodataset
            dataset.selected_classes = selected_classes
            distributed = sampler_options['images_per_batch'] > sampler_options['images_per_gpu']
            sampler = self.get_sampler(dataset, selected_classes, is_fewshot, is_support, sampler_options, distributed)
            batch_sampler = make_batch_data_sampler(
                dataset,
                sampler,
                sampler_options['aspect_grouping'],
                sampler_options['images_per_gpu'],
                sampler_options['num_iters'],
                sampler_options['start_iter'],
                is_fewshot=is_fewshot,
                is_support=is_support
                )

            data_loader = self.create_data_loader(dataset, batch_sampler)
            data_loaders.append(data_loader)

        self.build_categories_map(data_loaders[0].dataset)
        return data_loaders[0]

    def configure_sampler_options(self, is_fewshot):
        num_gpus = comm.get_world_size()
        images_per_batch = self.cfg.SOLVER.IMS_PER_BATCH if self.is_train else self.cfg.TEST.IMS_PER_BATCH
        images_per_gpu = math.ceil(images_per_batch / num_gpus) if self.is_train else images_per_batch # fix if multi gpu for eval

        return {
            'shuffle': True,
            'images_per_batch': images_per_batch,
            'images_per_gpu': images_per_gpu, 
            'start_iter': self.start_iter if self.is_train else 0,
            'num_iters': None if self.is_train and not is_fewshot else self.cfg.SOLVER.MAX_ITER,
            'aspect_grouping': [1] if self.cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
        }

    def get_sampler(self, dataset, selected_classes, is_fewshot, is_support, options, distributed_sampler=False):
        if is_fewshot:
            if is_support or self.is_finetune:
                n_query = self.cfg.FEWSHOT.K_SHOT
                if not self.cfg.FEWSHOT.SAME_SUPPORT_IN_BATCH and is_support:
                    # when same support in batch is deactivated
                    # one support should be sampled for each
                    # element of the batch
                    n_query *= options['images_per_gpu']

                self.select_examples(dataset, selected_classes, n_query)
                base_sampler = SupportSampler(dataset, self.selected_base_examples)
            else:
                n_query = self.cfg.FEWSHOT.N_QUERY_TRAIN if self.is_train else self.cfg.FEWSHOT.N_QUERY_TEST
                base_sampler = FilteringSampler(dataset, selected_classes, n_query, options['shuffle'], rng=self.rng_handler_fixed.torch_rng)
        else:
            base_sampler = FilteringSampler(dataset, selected_classes, len(dataset), options['shuffle'], rng=self.rng_handler_fixed.torch_rng)

        if distributed_sampler and self.is_train and not is_support : # tmp: eval is single gpu, check eval
            # Wrap the base sampler to respect the distributed indices
            base_sampler = samplers.DistributedIndexSampler(base_sampler)
        return base_sampler
    
    def select_examples(self, dataset, selected_classes, n_query):
        self.selected_base_examples, self.selected_novel_examples = \
            self.example_selector.select_examples(
                dataset,
                selected_classes,
                n_query, self.test_classes
                )

    def create_data_loader(self, dataset, batch_sampler):
        collator = BBoxAugCollator() if not self.is_train and self.cfg.TEST.BBOX_AUG.ENABLED else BatchCollator(self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        return DataLoader(
            dataset,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            worker_init_fn=self.rng_handler_fixed.worker_init_fn(),
            generator=self.rng_handler_fixed.torch_rng
            )
        
    def build_categories_map(self, coco_dataset):
        cats = coco_dataset.coco.cats
        cat_id_map = coco_dataset.json_category_id_to_contiguous_id
        self.categories = {}
        for k, v in cats.items():
            self.categories[cat_id_map[k]] = v['name']
