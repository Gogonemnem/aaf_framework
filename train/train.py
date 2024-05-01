import math
from tqdm import tqdm

import os
import datetime
import logging
import time
import sys

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm

from fcos.core.utils.metric_logger import MetricLogger

from ..modeling.detector import build_detection_model
from .utils import make_lr_scheduler, make_optimizer
from ..data.data_handler import DataHandler
from ..utils.checkpointer import DetectronCheckpointer
from ..eval import Evaluator
from ..utils.custom_logger import CustomLogger


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
            self.logger.info("Model:\n{}".format(self.model))

        self.episodes = cfg.FEWSHOT.EPISODES
        self.logging_int = cfg.LOGGING.INTERVAL
        self.logging_eval_int = cfg.LOGGING.EVAL_INTERVAL
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

        self.fintuning_start_iter = 0

        extra_checkpoint_data = self.checkpointer.load(self.cfg.MODEL.WEIGHT)
        self.arguments.update(extra_checkpoint_data)

        self.evaluator_test = None
        self.evaluator_train = None
    
    def setup_model(self):
        self.model = build_detection_model(self.cfg).to(self.device)

        self.distributed = comm.get_world_size() > 1
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

    def setup_environment(self):
        self.optimizer = make_optimizer(self.cfg, self.model)
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer)
        self.checkpointer = self.init_checkpointer()

        self.arguments = {'iteration': 0}
        self.is_finetuning = False
        # by default is_finetuning is false it will be set to true when it starts
        # if only finetuning then the number of base training is 0 but finetuning will
        # be set later anyway. 

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
        
        if not self.cfg.FINETUNING:
            return
        
        self.is_finetuning = True
        self.finetuning_start_iter = self.max_iter
        for k_shot in self.cfg.FINETUNE.SHOTS:
            # number of episodes specified in cfg finetune is for the
            # 1 shot case, number is adjusted to have the same number of
            # updates with each shots.
            episodes = self.calculate_episodes(k_shot)
            self.prepare_finetuning(k_shot, episodes)
            data_handler = DataHandler(self.cfg,
                                        base_classes=False,
                                        is_train=True,
                                        start_iter=self.arguments['iteration'],
                                        is_finetune=True)
            data_handler.task_sampler.display_classes()
            self.run_fs_training_loop(data_handler)
    
    def calculate_episodes(self, k_shot):
        """Calculate the number of training episodes based on configuration."""
        return self.cfg.FINETUNE.EPISODES // math.ceil(
            self.cfg.FEWSHOT.N_WAYS_TRAIN / self.cfg.SOLVER.IMS_PER_BATCH * k_shot
        )
    
    def prepare_finetuning(self, k_shot, episodes):
        self.episodes = episodes
        self.logging_int = self.cfg.LOGGING.INTERVAL // 1
        self.logging_eval_int = self.cfg.LOGGING.EVAL_INTERVAL // 3
        self.checkpoint_period = self.cfg.SOLVER.CHECKPOINT_PERIOD // 1

        self.checkpointer.load(os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth'))

        self.evaluator_train = None
        self.evaluator_test = None

        # Freeze backbone layer
        self.model.backbone.body._freeze_backbone(self.cfg.FINETUNE.FREEZE_AT)

        # Update optimizer (lr)
        del self.optimizer
        self.optimizer = make_optimizer(self.cfg, self.model, self.cfg.FINETUNE.LR)
        self.scheduler.milestones = [self.max_iter + s for s in self.cfg.FINETUNE.STEPS]

        # Update cfg
        self.cfg.merge_from_list(['FEWSHOT.K_SHOT', k_shot])

    def run_training_loop(self, data_handler, is_few_shot=False):
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
            query_loader, _, _ = data_handler.get_dataloader()
            iter_epoch = len(query_loader)
            self.max_iter = iter_epoch * self.episodes + start_iter
        else:
            data_loader = data_handler.get_dataloader()
            self.max_iter = len(data_loader)
        
        current_iter = start_iter
        steps_per_update = self.cfg.SOLVER.ACCUMULATION_STEPS
        accumulation_count = 0
        
        for epoch in range(self.episodes if is_few_shot else 1):
            if is_few_shot:
                # Reload dataloader for each few-shot episode
                query_loader, support_loader, train_classes = data_handler.get_dataloader(
                    seed=self.cfg.RANDOM.SEED if self.is_finetuning else None
                    )
                loader = query_loader
            else:
                train_classes = None
                loader = data_loader

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
                    if self.distributed:
                        support_features = self.model.module.compute_support_features(support_loader, self.device)
                    else:
                        support_features = self.model.compute_support_features(support_loader, self.device)

                accumulate = (accumulation_count + 1) % steps_per_update != 0

                losses, loss_dict = self.train_step(images, targets, classes=train_classes, support=support_features, accumulate=accumulate)

                accumulation_count += 1
                if accumulation_count % steps_per_update == 0:
                    accumulation_count = 0  # Reset accumulation count after reaching the accumulation steps

                if not comm.is_main_process():
                    continue

                batch_time = time.time() - end
                end = time.time()
                self.meters.update(time_batch=batch_time, time_data=data_time)

                last_iteration_reached = current_iter == self.max_iter

                log_interval_reached = current_iter % self.logging_int == 0
                should_log = log_interval_reached or last_iteration_reached
                if should_log:
                    self.log_metrics(losses, loss_dict, current_iter)
                
                eval_interval_reached = current_iter % self.logging_eval_int == 0
                should_eval = eval_interval_reached or last_iteration_reached
                if should_eval:
                    self.eval(current_iter, is_few_shot=is_few_shot)

                checkpoint_interval_reached = current_iter % self.checkpoint_period == 0
                if checkpoint_interval_reached:                    
                    self.save_checkpoint(f"intermediate_{current_iter:07d}", is_few_shot)

        
        if not comm.is_main_process():
            return
        
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total {} time: {} ({:.4f} s / it)".format(
                'base training' if not self.is_finetuning else 'finetuning',
                total_time_str, total_training_time / (self.max_iter + 1)
            )
        )
        self.save_checkpoint("final_model", is_few_shot)

    def train_step(self, images, targets, classes=None, support=None, accumulate=False):
        """A single training step."""
        images = images.to(self.device)
        targets = [target.to(self.device) for target in targets]
        loss_dict = self.model(images, targets, classes=classes, support=support)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = comm.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        losses_reduced /= self.cfg.SOLVER.ACCUMULATION_STEPS
        losses_reduced.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) # use only for MFRCN to reduce unstability

        if not accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
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
        self.model.eval()

        # Initialize or use existing evaluator for training classes
        if self.evaluator_train is None:
            self.evaluator_train = self.create_evaluator(is_train_class=True, is_few_shot=is_few_shot)

        res_train = self.evaluator_train.eval(verbose=False, per_category=False)['overall']
        train_map = res_train.stats[1] if res_train != {} else 0

        if is_few_shot:
            # Initialize or use existing evaluator for test classes only if in few-shot mode
            if self.evaluator_test is None:
                self.evaluator_test = self.create_evaluator(is_train_class=False, is_few_shot=is_few_shot)
            
            # Perform evaluation and retrieve results for test classes
            res_test = self.evaluator_test.eval(verbose=False, per_category=False)['overall']
            test_map = res_test.stats[1] if res_test != {} else 0

            eval_res = {'Train mAP': train_map, 'Test mAP': test_map}
        else:
            eval_res = {'Train mAP': train_map}

        if comm.is_main_process():
            self.tensorboard.add_multi_scalars(eval_res, iteration, main_tag='Eval')
        self.model.train()
    
    def create_evaluator(self, is_train_class, is_few_shot):
        """
        Create an evaluator based on the class type and whether it is few-shot training.
        """
        base_classes = is_train_class or not is_few_shot
        data_handler = DataHandler(self.cfg, base_classes=base_classes, data_source='val', is_train=False)
        return Evaluator(self.model, self.cfg, data_handler)
    
    def save_checkpoint(self, model_name, is_few_shot=False):
        if is_few_shot:
            model_name = f"{model_name}_{self.cfg.FEWSHOT.K_SHOT}shot"
            if self.is_finetuning:
                model_name = f"{model_name}_finetuning"
        else:
            model_name = f"{model_name}_base"

        self.checkpointer.save(model_name, **self.arguments)

    