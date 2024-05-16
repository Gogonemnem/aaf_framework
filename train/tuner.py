import optuna
from detectron2.config import get_cfg

from .train import Trainer
from ..config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)

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



def get_cfg_with_hyperparams(args, lr, accum_steps, weight_decay, episodes):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.ACCUMULATION_STEPS = accum_steps
    cfg.FEWSHOT.EPISODES = episodes
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000000000
    cfg.LOGGING.EVAL_INTERVAL = 1000000000

    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + f"_lr={lr}_accum={accum_steps}_wd={weight_decay}"
    cfg.freeze()
    return cfg

def objective_first_stage(args, trial):
    # Define hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    accumulation_steps = trial.suggest_int('accumulation_steps', 0, 4)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    # Get modified config
    cfg = get_cfg_with_hyperparams(args, lr, accumulation_steps, weight_decay, 10)
    default_setup(cfg, None)

    # Create the trainer
    trainer = Trainer(cfg)

    # Train for fewer epochs
    loss = trainer.train()

    return loss

def objective_second_stage(args, trial):
    # Load hyperparameters from one of the best trials
    params = trial.user_attrs['params']
    
    lr = params['lr']
    accumulation_steps = params['accumulation_steps']
    weight_decay = params['weight_decay']

    # Set the number of episodes for the second stage
    num_full_episodes = 100  # Adjust this as needed

    # Get modified config
    cfg = get_cfg_with_hyperparams(args, lr, accumulation_steps, weight_decay, num_full_episodes)
    default_setup(cfg, None)

    # Create the trainer
    trainer = Trainer(cfg)

    # Train for fewer epochs
    loss = trainer.train()

    return loss
