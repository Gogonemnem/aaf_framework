import os
from .train.train import Trainer
from .config import cfg as config
import sys
import torch
from torch import cuda
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)

def set_cuda_visible_devices():
    cuda_visible_devices = ','.join(str(i) for i in range(cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = config.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(
    #     cfg, args
    # )
    return cfg
    

def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    
    if not args.eval_only:
        trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
    
