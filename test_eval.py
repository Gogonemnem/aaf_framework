
from .config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)
import json
import os

from .data.data_handler import DataHandler
from .eval.eval import Evaluator
from .modeling.detector.detectors import build_detection_model
from .utils.checkpointer import DetectronCheckpointer
from .modeling.detector import build_detection_model
from .train.utils import make_lr_scheduler, make_optimizer


def eval_model(cfg, model, is_few_shot=False, k_shot=None):
    """
    Perform evaluation, handling both few-shot and regular scenarios based on the is_few_shot flag.
    """
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    
    # Set the correct checkpoint file name
    if is_few_shot:
        checkpoint_file = f"final_model_{k_shot}shot_finetuning.pth"
    else:
        checkpoint_file = "final_model_1shot.pth"
    
    checkpoint_path = os.path.join(output_dir, checkpoint_file)
    
    if os.path.exists(checkpoint_path):
        DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk=False
        ).load(checkpoint_path, force_file=True)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"No checkpoint found at: {checkpoint_path}")
        return {}

    model.eval()
    evaluator = create_evaluator(model, cfg, is_train_class=True, is_few_shot=is_few_shot)

    eval_res = evaluator.eval_all(n_episode=100, verbose=True)
    return eval_res


def create_evaluator(model, cfg, is_train_class, is_few_shot):
    """
    Create an evaluator based on the class type and whether it is few-shot training.
    """
    base_classes = is_train_class or not is_few_shot
    # data_handler = DataHandler(cfg, base_classes=base_classes, data_source='test', is_train=False)
    data_handler = DataHandler(cfg, base_classes=base_classes, data_source='test', eval_all=True, is_train=False)
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
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}
    cfg = setup(args)
    model = build_detection_model(cfg).to("cuda")

    config_basename = os.path.basename(args.config_file)
    
    # Evaluate base training model
    if args.overwrite or "base" not in results.get(config_basename, {}):
        base_res = eval_model(cfg, model, is_few_shot=True)
        results.update({config_basename: {"base": base_res}})
    
    # Evaluate few-shot fine-tuning models
    few_shot_results = results[config_basename].get("few_shot", {})
    for k_shot in cfg.FINETUNE.SHOTS:
        shot_key = f"{k_shot}_shot"
        if not args.overwrite and shot_key in few_shot_results:
            print(f"Skipping evaluation for {shot_key} as results already exist.")
            continue
        shot_res = eval_model(cfg, model, is_few_shot=True, k_shot=k_shot)
        few_shot_results[shot_key] = shot_res

    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {} 
    results.update({config_basename: {"few_shot": few_shot_results}})
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f)

def invoke_main() -> None:
    parser = default_argument_parser()
    parser.add_argument("--output-file", required=True, help="Output file for evaluation results")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Flag to overwrite existing results in the output file")
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
