import datetime
import logging
import math
import os
import sys
import time

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.utils import comm
from fcos.core.utils.metric_logger import MetricLogger
from fcos.core.utils.checkpoint import DetectronCheckpointer

from ..data.data_handler import DataHandler
from ..eval import Evaluator
from ..modeling.detector import build_detection_model
from ..utils.checkpointer import DetectronCheckpointer
from ..utils.custom_logger import CustomLogger
from .utils import make_lr_scheduler, make_optimizer


class Trainer():
    """
    Trainer object that manages the training of all networks no matter
    if it few-shot or not (or finetuning)

    Builds network and environment from cfg file. 
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.setup_model()
        self.setup_environment()
        if comm.is_main_process():
            self.setup_logging()

        
        self.logging_int = cfg.LOGGING.INTERVAL
        self.logging_eval_int = cfg.LOGGING.EVAL_INTERVAL
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    def setup_model(self, k_shot=None):
        if k_shot is not None:
            # Update cfg entry for k'th shot
            self.cfg.merge_from_list(['FEWSHOT.K_SHOT', k_shot])

        self.model = build_detection_model(self.cfg).to(self.device)

        self.distributed = comm.get_world_size() > 1
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
                find_unused_parameters=True
            )

    def setup_environment(self, is_finetuning=False, k_shot=1):
        self.episodes = self.calculate_episodes(k_shot)

        lr = None
        if is_finetuning:
            lr = self.cfg.FINETUNE.LR
        self.optimizer = make_optimizer(self.cfg, self.model, lr)


        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer)
        if is_finetuning:
            self.scheduler.milestones = [self.max_iter + s for s in self.cfg.FINETUNE.STEPS]

        self.checkpointer = self.init_checkpointer()

        # Load checkpoint data
        extra_checkpoint_data = self.checkpointer.load(self.cfg.MODEL.WEIGHT)
        self.arguments = extra_checkpoint_data

        batch_size = self.cfg.SOLVER.IMS_PER_BATCH

        # Assuming the original batch size and accumulation steps are stored in checkpoint data
        original_batch_size = extra_checkpoint_data.get('prior_batch_size', batch_size)

        # Update arguments dictionary with new values
        self.arguments.update({
            'prior_batch_size': batch_size,
        })

        # If the checkpoint contains iteration information, adjust it based on the batch size and accumulation steps
        if 'iteration' in extra_checkpoint_data:
            # Recalculate start_iter based on the new configuration
            self.arguments['iteration'] = int(extra_checkpoint_data['iteration'] * (original_batch_size / batch_size))
        else:
            self.arguments['iteration'] = 0

        self.is_finetuning = is_finetuning
        self.finetuning_start_iter = 0 if not is_finetuning else self.base_max_iter
        # by default is_finetuning is false it will be set to true when it starts
        # if only finetuning then the number of base training is 0 but finetuning will
        # be set later anyway.

        self.evaluator_test = None
        self.evaluator_train = None

        if is_finetuning:
             # Freeze backbone layer
            model = self.model if not self.distributed else self.model.module
            model.backbone.body._freeze_backbone(self.cfg.FINETUNE.FREEZE_AT)

    def init_checkpointer(self):
        """Initialize the model checkpointer."""
        output_dir = self.cfg.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        return DetectronCheckpointer(
            self.cfg, self.model, self.optimizer, self.scheduler, output_dir, save_to_disk=True
        )
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console Handler for printing the logs to the console
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(log_format)
        # self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(self.cfg.OUTPUT_DIR, 'training.log'))
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Model:\n{self.model}")

        self.tensorboard = CustomLogger(log_dir=os.path.join(self.cfg.OUTPUT_DIR, 'logs'), notify=False)

    def train(self):
        """
        Main training loop. Starts base training and multiple finetunings
        with different number of shots after (if finetuning is enabled). 
        """
        if self.cfg.FEWSHOT.ENABLED:
            self.train_few_shot()
        else:
            self.train_base()

    def train_base(self):
        """Perform base training."""
        data_handler = DataHandler(self.cfg, base_classes=True, is_train=True,
                                   start_iter=self.arguments['iteration'])
        self.run_training_loop(data_handler)

    def train_few_shot(self):
        """Perform few-shot training and fine-tuning as configured."""
        data_handler = DataHandler(self.cfg, base_classes=True, is_train=True,
                                   start_iter=self.arguments['iteration'])
        data_handler.task_sampler.display_classes()
        self.run_training_loop(data_handler, is_few_shot=True)
        comm.synchronize()

        if not self.cfg.FINETUNING:
            return

        for k_shot in self.cfg.FINETUNE.SHOTS:
            # number of episodes specified in cfg finetune is for the
            # 1 shot case, number is adjusted to have the same number of
            # updates with each shots.
            self.setup_model(k_shot=k_shot)
            self.setup_environment(is_finetuning=True, k_shot=k_shot)

            if self.final_checkpoint_exists(is_few_shot=True):
                if comm.is_main_process():
                    self.logger.info(f"Final checkpoint for {k_shot}-shot already exists. Skipping finetuning.")
                continue

            data_handler = DataHandler(
                self.cfg,
                base_classes=False,
                is_train=True,
                start_iter=self.arguments['iteration'],
                is_finetune=True
                )
            data_handler.task_sampler.display_classes()
            self.run_training_loop(data_handler, is_few_shot=True)
            comm.synchronize()
    
    def calculate_episodes(self, k_shot):
        """Calculate the number of training episodes based on configuration."""
        # I have doubts regarding the result when batches_per_episode is larger than 1,
        # Is the data handler consistent with the 'next' episode?
        effective_batch_size = self.cfg.SOLVER.IMS_PER_BATCH * self.cfg.SOLVER.ACCUMULATION_STEPS
        batches_per_episode = math.ceil(self.cfg.FEWSHOT.N_WAYS_TRAIN * k_shot / effective_batch_size)
        return self.cfg.FINETUNE.EPISODES // batches_per_episode

    def run_training_loop(self, data_handler: DataHandler, is_few_shot=False):
        """
        Training loop for both base and few-shot training.
        """
        start_iter = self.arguments["iteration"]
        self.model.train()

        if comm.is_main_process():
            self.logger.info("Start training")
            self.meters = MetricLogger(delimiter="  ")
            start_training_time = time.time()
            end = time.time()

        if is_few_shot:
            query_loader, _, _ = data_handler.get_dataloader(
                seed=self.cfg.RANDOM.SEED if self.is_finetuning else None
            )
            iter_epoch = len(query_loader)
            self.max_iter = iter_epoch * self.episodes + self.finetuning_start_iter
            if not self.is_finetuning:
                self.base_max_iter = iter_epoch * self.episodes
        else:
            data_loader = data_handler.get_dataloader()
            iter_epoch = len(data_loader)
            self.max_iter = iter_epoch + self.finetuning_start_iter
        
        current_iter = 0 if not self.is_finetuning else self.finetuning_start_iter
        steps_per_update = self.cfg.SOLVER.ACCUMULATION_STEPS
        accumulation_count = 0
        
        for epoch in range(self.episodes if is_few_shot else 1):
            if is_few_shot:
                # Reload dataloader for each few-shot episode
                loader, support_loader, train_classes = data_handler.get_dataloader(
                    seed=self.cfg.RANDOM.SEED if self.is_finetuning else None
                    )
            else:
                train_classes = None
                support_loader = None
                loader = data_handler.get_dataloader()

            if current_iter + iter_epoch <= start_iter:
                current_iter += iter_epoch
                continue

            if comm.is_main_process():
                loader = tqdm(loader)

                if is_few_shot:
                    self.logger.info(f'Episode {epoch + 1}: classes = {train_classes}')

            for images, targets, _ in loader:
                current_iter += 1
                if current_iter < start_iter:
                    continue

                self.arguments["iteration"] = current_iter

                if comm.is_main_process():
                    data_time = time.time() - end

                support_features = None
                if is_few_shot:
                    model = self.model if not self.distributed else self.model.module
                    support_features = model.compute_support_features(support_loader, self.device)

                accumulate = (accumulation_count + 1) % steps_per_update != 0

                losses, loss_dict = self.train_step(images, targets, classes=train_classes, support=support_features, accumulate=accumulate)

                accumulation_count += 1
                if accumulation_count % steps_per_update == 0:
                    accumulation_count = 0  # Reset accumulation count after reaching the accumulation steps
                
                if not comm.is_main_process():
                    continue

                last_iteration_reached = current_iter == self.max_iter

                ## tmp: eval on 1 gpu, because of issues with gathering the predictions: so change data_handler sampler (and options) & eval
                eval_interval_reached = current_iter % self.logging_eval_int == 0
                should_eval = eval_interval_reached or last_iteration_reached
                if should_eval:
                    self.eval(current_iter, is_few_shot=is_few_shot)

                batch_time = time.time() - end
                end = time.time()
                self.meters.update(time_batch=batch_time, time_data=data_time)

                log_interval_reached = current_iter % self.logging_int == 0
                should_log = log_interval_reached or last_iteration_reached
                if should_log:
                    self.log_metrics(losses, loss_dict, current_iter)

                checkpoint_interval_reached = current_iter % self.checkpoint_period == 0
                if checkpoint_interval_reached:
                    self.save_checkpoint(f"intermediate_{current_iter:07d}", is_few_shot)
            self.logger.info(targets)
        if not comm.is_main_process():
            return

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total %s time: %s (%.4f s / it)",
            'base training' if not self.is_finetuning else 'finetuning',
            total_time_str,
            total_training_time / (self.max_iter + 1)
            )
        self.save_checkpoint("final_model", is_few_shot)

    def train_step(self, images, targets, classes=None, support=None, accumulate=False):
        """A single training step."""
        images = images.to(self.device)
        targets = [target.to(self.device) for target in targets]
        loss_dict = self.model(images, targets, classes=classes, support=support)
        losses = sum(loss for loss in loss_dict.values())

        losses /= self.cfg.SOLVER.ACCUMULATION_STEPS
        losses.backward()

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) # use only for MFRCN to reduce unstability

        if not accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = comm.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        return losses_reduced, loss_dict_reduced

    def log_metrics(self, losses, loss_dict, iteration):
        """Log training metrics to console and tensorboard."""
        eta_seconds = self.meters.time_batch.global_avg * (self.max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        self.meters.update(loss=losses, **loss_dict)

        self.logger.info(
            self.meters.delimiter.join(
                [
                    f"eta: {eta_string}",
                    f"iter: {iteration}/{self.max_iter}",
                    f"{str(self.meters)}",
                    f"lr: {self.optimizer.param_groups[0]['lr']:.6f}",
                    f"max mem: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}",
                ]
            )
        )
        if sys.gettrace() is None:
            self.tensorboard.add_multi_scalars(self.meters.meters, iteration)

    def eval(self, iteration, is_few_shot=False):
        """
        Perform evaluation, handling both few-shot and regular scenarios based on the is_few_shot flag.
        """
        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        self.model.eval()

        # Initialize or use existing evaluator for training classes
        if self.evaluator_train is None:
            self.evaluator_train = self.create_evaluator(is_train_class=True, is_few_shot=is_few_shot)

        res_train = self.evaluator_train.eval(verbose=False, per_category=False, seed=self.cfg.RANDOM.SEED).get('overall', {})
        metrics_dict = {f"Train {metric}": res_train.stats[id] if res_train != {} else 0 for id, metric in enumerate(metrics)}

        if is_few_shot:
            # Initialize or use existing evaluator for test classes only if in few-shot mode
            if self.evaluator_test is None:
                self.evaluator_test = self.create_evaluator(is_train_class=False, is_few_shot=is_few_shot)

            # Perform evaluation and retrieve results for test classes
            res_test = self.evaluator_test.eval(verbose=False, per_category=False, seed=self.cfg.RANDOM.SEED).get('overall', {})
            metrics_dict.update({f"Test {metric}": res_test.stats[id] if res_test != {} else 0 for id, metric in enumerate(metrics)})

        if comm.is_main_process():
            self.tensorboard.add_multi_scalars(metrics_dict, iteration, main_tag='Eval')
        self.model.train()

    def create_evaluator(self, is_train_class, is_few_shot):
        """
        Create an evaluator based on the class type and whether it is few-shot training.
        """
        base_classes = is_train_class or not is_few_shot
        data_handler = DataHandler(self.cfg, base_classes=base_classes, data_source='val', is_train=False, seed=self.cfg.RANDOM.SEED)
        return Evaluator(self.model, self.cfg, data_handler)

    def final_checkpoint_exists(self, is_few_shot=False):
        name = self.generate_checkpoint_name('final_model', is_few_shot)
        return os.path.exists(os.path.join(self.cfg.OUTPUT_DIR, f"{name}.pth"))
    
    def save_checkpoint(self, model_name, is_few_shot=False):
        name = self.generate_checkpoint_name(model_name, is_few_shot)

        self.checkpointer.save(name, **self.arguments)
        if model_name == 'final_model':
            if is_few_shot:
                path = os.path.join(self.cfg.OUTPUT_DIR, f"{model_name}_1shot.pth") # hardcoded 1 shot for now
                self.checkpointer.tag_last_checkpoint(path)

    def generate_checkpoint_name(self, model_name, is_few_shot=False):
        if is_few_shot:
            name = f"{model_name}_{self.cfg.FEWSHOT.K_SHOT}shot"
            if self.is_finetuning:
                name = f"{name}_finetuning"
        else:
            name = f"{model_name}_base"
        return name
