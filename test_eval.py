
from .config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)
import os

from .data.data_handler import DataHandler
from .eval.eval import Evaluator
from .modeling.detector.detectors import build_detection_model
from .utils.checkpointer import DetectronCheckpointer
from .modeling.detector import build_detection_model
from .train.utils import make_lr_scheduler, make_optimizer


def eval(model, cfg, is_few_shot=False):
    """
    Perform evaluation, handling both few-shot and regular scenarios based on the is_few_shot flag.
    """
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    last_checkpoint =  DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk=False
    ).load()

    if last_checkpoint:
        print(f"Resumed from checkpoint")
    else:
        print("No valid checkpoint found, training from scratch.")

    model.eval()
    evaluator_train = create_evaluator(model, cfg, is_train_class=True, is_few_shot=is_few_shot)

    res_train = evaluator_train.eval(verbose=True, per_category=True)['overall']
    train_map = res_train.stats[1] if res_train != {} else 0

    if is_few_shot:
        evaluator_test = create_evaluator(model, cfg, is_train_class=False, is_few_shot=is_few_shot)
        
        # Perform evaluation and retrieve results for test classes
        res_test = evaluator_test.eval(verbose=True, per_category=False)['overall']
        test_map = res_test.stats[1] if res_test != {} else 0

        eval_res = {'Train mAP': train_map, 'Test mAP': test_map}
    else:
        eval_res = {'Train mAP': train_map}
    
    model.train()
    return eval_res


def create_evaluator(model, cfg, is_train_class, is_few_shot):
    """
    Create an evaluator based on the class type and whether it is few-shot training.
    """
    base_classes = is_train_class or not is_few_shot
    data_handler = DataHandler(cfg, base_classes=base_classes, data_source='val', is_train=False)
    return Evaluator(model, cfg, data_handler)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_detection_model(cfg).to("cuda")
    eval(model, cfg, is_few_shot=True)

def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
