import torch
import os
from contextlib import nullcontext

import numpy as np

import detectron2.utils.comm as comm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from tqdm import tqdm
import json

# dirty hack to remove prints from pycocotools
from ..utils import HiddenPrints


class Evaluator():
    """ 
    Evaluation class. 
    Performs evalution on a given model and DataHandler. 

    Two main functions can be used: 
        - eval: which runs evaluation on the DataHandler query set.
        - eval_all: which use DataHandler instantiated with eval_all=True. 
                    It runs eval on all the loaders provided by the DataHandler.  
    
    Arguments:
        - model: pytorch model to be evaluated.
        - cfg: cfg object of the model.
        - data_handler: DataHandler object of the dataset.
    """
    def __init__(self, model, cfg, data_handler):
        self.model = model
        self.cfg = cfg
        self.data_handler = data_handler
        self.device = cfg.MODEL.DEVICE
        self.output_folder = self.cfg.OUTPUT_DIR
        self.categories =  None
        self.current_classes = None

    def eval(self, verbose=True, per_category=True, loaders=None, seed=None):
        """
        Eval function on a single data loader (or couple query/support loaders)

        Arguments:
            - verbose: print eval results at the end of computation.
            - all_classes:  
        """
        if seed is not None:
            self.data_handler.rng_handler.update_seeds(seed)

        loaders = loaders or self.data_handler.get_dataloader(seed=seed)
        predictions = self.collect_model_predictions(loaders)
        has_predictions = sum(len(pred) for pred in predictions) > 0

        if not has_predictions:
            return {}
        
        context_manager = HiddenPrints() if not verbose else nullcontext()
        with context_manager:
            dataset = loaders[0].dataset if isinstance(loaders, tuple) else loaders.dataset
            self.save_coco_results(predictions, dataset)
            results = self.perform_coco_evaluation(dataset, verbose, per_category)
            return results

    def collect_model_predictions(self, loaders, verbose=False):
        query_loader, support_loader = loaders[:2] if self.cfg.FEWSHOT.ENABLED else (loaders, None)
        classes = loaders[-1] if self.cfg.FEWSHOT.ENABLED else np.array(list(query_loader.dataset.coco.cats.keys())) + 1

        self.current_classes = classes
        if verbose:
            print('Evaluation on classes: {}'.format(str(classes)))

        self.categories = {
                idx: v['name']
                for idx, v in query_loader.dataset.coco.cats.items()
            }
        self.contiguous_label_map = query_loader.dataset.contiguous_category_id_to_json_id
            
        predictions = self.predict(query_loader, support_loader, classes)
        for pred in predictions:
            pred.add_field("objectness", torch.ones(len(pred), device=self.device))
        return predictions
    
    def save_coco_results(self, predictions, dataset):
        coco_results = self.prepare_for_coco_detection(predictions, dataset)

        json_result_file = os.path.join(self.output_folder, 'coco_results.json')
        with open(json_result_file, 'w') as f:
            json.dump(coco_results, f)
    
    def prepare_for_coco_detection(self, predictions, dataset):
        """
        Convert predictions from model into coco format detections.
        """
        coco_results = []
        for prediction in predictions:
            if len(prediction) > 0:
                formatted_predictions = self.format_coco_detection(prediction, dataset)
                coco_results.extend(formatted_predictions)
        return coco_results

    def format_coco_detection(self, prediction, dataset):
        """
        Format a single prediction into COCO detection format, adjusting for image dimensions.
        """
        image_id = prediction.get_field('image_id').item()
        img_info = dataset.get_img_info(image_id)
        original_id = dataset.id_to_img_map[image_id]
        prediction = prediction.resize((img_info["width"], img_info["height"]))
        prediction = prediction.convert("xywh")
        
        formatted_predictions = []
        for box, score, label in zip(prediction.bbox.tolist(), prediction.get_field("scores").tolist(), prediction.get_field("labels").tolist()):
            formatted_predictions.append({
                "image_id": original_id,
                "category_id": dataset.contiguous_category_id_to_json_id[label],
                "bbox": box,
                "score": score,
            })
        return formatted_predictions
    
    def perform_coco_evaluation(self, dataset, verbose=True, per_category=False):
        """
        Run coco evaluation using pycocotools. 
        """
        sep = f'+{"-"*77}+'
        
        json_result_file = os.path.join(self.output_folder, 'coco_results.json')
        coco_gt = dataset.coco
        coco_dt = coco_gt.loadRes(json_result_file)

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [dataset.contiguous_category_id_to_json_id[c] for c in self.current_classes]
        coco_eval.params.imgIds = list(set([det['image_id'] for det in list(coco_dt.anns.values())]))
        coco_eval.evaluate()
        coco_eval.accumulate()

        print(f'{sep}')
        print('\tOverall results:')
        print(f'{sep}')
        coco_eval.summarize()

        results = {'overall': coco_eval}
        
        if not per_category:
            return results

        for category_id in self.current_classes:
            coco_eval.params.catIds = [dataset.contiguous_category_id_to_json_id[category_id]]
            coco_eval.params.imgIds = list(set([det['image_id'] for det in list(coco_dt.anns.values())]))
            coco_eval.evaluate()
            coco_eval.accumulate()

            print(f'{sep}')
            print(f'\tResults for class {self.categories[category_id]}:')
            print(f'{sep}')
            coco_eval.summarize()

            results[self.categories[category_id]] = coco_eval
        return results
        
        

    def predict(self, query_loader, support_loader=None, classes=None):
        """
        Model inference on a query_loader without fewshot. 
        """
        predictions = []
        with torch.no_grad():
            for (images, targ, img_id) in tqdm(query_loader):
                model = self.model

                if comm.get_world_size() > 1:
                    model = model.module

                if support_loader is not None and classes is not None:
                    support = model.compute_support_features(support_loader, self.device)
                else:
                    support = None

                images = images.to(self.device)
                pred_batch = self.model(images, classes=classes, support=support)
                for idx, pred in enumerate(pred_batch):
                    pred.add_field('image_id', torch.tensor(img_id[idx]))  # store img_id as tensor for convenience
                    predictions.append(pred.to('cpu'))

        return predictions


    ### From here on, i dont know where it gets called.
    def eval_all(self, n_episode=1, verbose=True, seed=None):
        """
        Similar to eval function except it loop over the multiple dataloaders returned 
        by the DataHandler. (DataHandler must have eval_all=True).

        Results are then accumulated and stored in a pandas dataframe. 
        
        """
        assert self.data_handler.eval_all == True, 'Use eval_all with eval_all=True in DataHandler'
        accumulated_res_test = {}
        accumulated_res_train = {}
        all_res = {
            'train': accumulated_res_test,
            'test': accumulated_res_train
        }

        for eval_ep in range(n_episode):
            if seed is not None:
                self.data_handler.rng_handler.update_seeds(seed)
            loaders = self.data_handler.get_dataloader(seed=seed)
            for setup in ['train', 'test']:
                res_all_cls = {}
                for q_s_loaders in loaders[setup]:
                    _, res_cls = self.eval(verbose=False, loaders=q_s_loaders, seed=seed)
                    # this will overwrite some keys if the last batch is padded
                    # but only one eval is retained for each class
                    res_all_cls.update(res_cls)

                for k, v in res_all_cls.items():
                    if not k in all_res[setup]:
                        all_res[setup][k] = []
                    all_res[setup][k].append(v.stats)

        for setup in ['train', 'test']:
            for k, v in all_res[setup].items():
                all_res[setup][k] = np.vstack(all_res[setup][k]).mean(axis=0)

        return self.prettify_results(all_res, verbose=verbose, is_few_shot=True)

    def eval_no_fs(self, seed=None, verbose=False):
        """
        Eval without fewshot.  
        """
        overall_res, res_per_class = self.eval(seed=seed, verbose=verbose)
        for k in res_per_class:
            res_per_class[k] = res_per_class[k].stats

        return self.prettify_results(res_per_class, verbose=verbose, is_few_shot=False)

    def ignore_dataset_annot_without_pred(self, coco_gt, coco_dt, classes):
        img_with_predictions = set([det['image_id'] for det in list(coco_dt.anns.values())])
        gt_anns = coco_gt.anns
        classes_json = [self.contiguous_label_map[c] for c in classes]
        rm_keys = []
        for k, v in gt_anns.items():
            if v['image_id'] not in img_with_predictions or \
                (classes is not None and v['category_id'] not in classes_json):
                # category id is not necesarily contiguous
                gt_anns[k]['ignore'] = 1
                rm_keys.append(k)
            elif v['image_id'] not in img_with_predictions:
                del coco_gt.imgs[v['image_id']]

        for k in rm_keys:
            del gt_anns[k]

    def prettify_results(self, results, verbose=True, is_few_shot=False):
        """
        Prettify method build pandas dataframes from results of evaluation. 
        """
        import pandas as pd
        metrics = {}

        metrics['Measure'] = ['AP'] * 6 + ['AR'] * 6
        metrics['IoU'] = [
            '0.50:0.95',
            '0.50',
            '0.75',
        ] + ['0.50:0.95'] * 9
        metrics['Area'] = ['all', 'all', 'all', 'small', 'medium', 'large'] * 2

        df_metrics = pd.DataFrame.from_dict(metrics)

        if is_few_shot:
            results_train = results['train']
        else:
            results_train = results
        df_train = pd.DataFrame.from_dict(results_train)
        df_train = df_train.reindex(sorted(df_train.columns), axis=1)

        if is_few_shot:
            df_test = pd.DataFrame.from_dict(results['test'])
            df_test = df_test.reindex(sorted(df_test.columns), axis=1)

        list_all = [df_metrics, df_train]
        if is_few_shot:
            list_all.append(df_test)

        df_all = pd.concat(list_all, axis=1)
        df_all = df_all.set_index(['Measure', 'IoU', 'Area'])

        columns = [('Train classes', c) if c in results_train.keys() else
                   ('Test classes', c) for c in df_all.columns]
        df_all.columns = pd.MultiIndex.from_tuples(columns)

        if verbose:
            print(df_all)
        return df_all
