import optuna

from .train.tuner import objective_first_stage, objective_second_stage

from detectron2.engine import (
    default_argument_parser,
)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    # Define the storage for Optuna studies
    storage = "sqlite:///optuna_study.db"

    # First stage: Test 50 configurations with 10 episodes each
    first_stage_study = optuna.create_study(direction='minimize', study_name="first_stage_study", storage=storage, load_if_exists=True)
    first_stage_study.optimize(lambda trial: objective_first_stage(args, trial), n_trials=100)  # Adjust the number of trials and timeout as needed

    # Get the best top_k hyperparameter sets from the first stage

    top_k = 10
    best_trials = first_stage_study.best_trials[:top_k]
    
    print("Best hyperparameters from first stage: ", [trial.params for trial in best_trials])
    print("Best validation loss from first stage: ", first_stage_study.best_value)

    # Prepare to run the second stage
    second_stage_study = optuna.create_study(direction='minimize')

    # Add the best trials to the second stage study
    for trial in best_trials:
        second_stage_study.enqueue_trial(trial.params)
    
    # Run the second stage optimization
    second_stage_study.optimize(lambda trial: objective_second_stage(args, trial), n_trials=top_k)

     # Print the best hyperparameters from the second stage
    print("Best hyperparameters from second stage: ", second_stage_study.best_params)
    print("Best validation loss from second stage: ", second_stage_study.best_value)


    # Full training with the best hyperparameters
    # best_params = study.best_params
    # full_cfg = get_cfg_with_hyperparams(best_params['lr'], best_params['batch_size'], best_params['weight_decay'], num_warmup_episodes=None)  # Use full training setup
    # default_setup(full_cfg, None)
    
    # # Create the trainer with the best hyperparameters
    # full_trainer = Trainer(full_cfg)
    # full_trainer.train_base()  # Train with the full number of episodes
