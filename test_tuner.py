import optuna

from .train.tuner import objective_first_stage, objective_second_stage

from detectron2.engine import (
    default_argument_parser,
)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    first_stage_study = optuna.create_study(direction='minimize')
    first_stage_study.optimize(lambda trial: objective_first_stage(args, trial), n_trials=50)  # Adjust the number of trials and timeout as needed

    # Get the best 20 hyperparameter sets from the first stage
    best_trials = first_stage_study.best_trials[:10]
    
    print("Best hyperparameters: ", first_stage_study.best_params)
    print("Best validation loss: ", first_stage_study.best_value)

    # Prepare to run the second stage
    second_stage_study = optuna.create_study(direction='minimize')

    # Add the best trials to the second stage study
    for trial in best_trials:
        second_stage_study.enqueue_trial(trial.s)
    
    # Run the second stage optimization
    second_stage_study.optimize(objective_second_stage, n_trials=20)

    # Print the best hyperparameters from the second stage
    print("Best hyperparameters: ", second_stage_study.best_params)
    print("Best validation loss: ", second_stage_study.best_value)

    # Full training with the best hyperparameters
    # best_params = study.best_params
    # full_cfg = get_cfg_with_hyperparams(best_params['lr'], best_params['batch_size'], best_params['weight_decay'], num_warmup_episodes=None)  # Use full training setup
    # default_setup(full_cfg, None)
    
    # # Create the trainer with the best hyperparameters
    # full_trainer = Trainer(full_cfg)
    # full_trainer.train_base()  # Train with the full number of episodes
