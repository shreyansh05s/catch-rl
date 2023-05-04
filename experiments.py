import wandb
import Reinforce
import ActorCritic
import argparse
from copy import deepcopy

# default hyperparameters
hyperparameter_defaults = {
    "lr": 1e-2,
    "gamma": 0.85,
    "n_steps": 40,
    "entropy_weight": 0.01,
    "entropy_regularization": True,
    "bootstrap": False,
    "baseline": False,
    "normalize_returns": True,
    "lr_decay": 0.99,
    "lr_step_size": 50
}

# define different experiments in a list
global_exp_params = {
    "wandb_project": "catch-rl-experiments",
    "render": False,
    "num_episodes": 100,
    "log_interval": 10,
    "experiment": True,
    "verbose": False,
}

experiments = [
    {
        "name": "Baseline",
        "group": "Baseline",
        "job_type": "A2C",
        "model": ActorCritic,
        "observation_type": "pixel",
        "env": "default",
    },
    {
        "name": "Baseline",
        "group": "Baseline",
        "job_type": "A2C-Bootstrap",
        "model": ActorCritic,
        "observation_type": "pixel",
        "env": "default",
        "hyperparameter": {
            "bootstrap": True,
            "n_steps": 40
        }
    },
    {
        "name": "Baseline",
        "group": "Baseline",
        "job_type": "A2C-Baseline",
        "model": ActorCritic,
        "observation_type": "pixel",
        "env": "default",
        "hyperparameter": {
            "baseline": True
        }
    },
    {
        "name": "Baseline",
        "group": "Baseline",
        "job_type": "A2C-Bootstrap-Baseline",
        "model": ActorCritic,
        "observation_type": "pixel",
        "env": "default",
        "hyperparameter": {
            "baseline": True,
            "bootstrap": True,
            "n_steps": 40
        }
    },
]




default_env = {
    "rows": 7,
    "columns": 7,
    "speed": 1.0,
    "max_steps": 250,
    "max_misses": 10,
    "observation_type": "pixel",
}

number_of_repeats = 2

for exp in experiments:
    # run each experiment 10 times

    params = deepcopy(hyperparameter_defaults)
    if "hyperparameter" in exp:
        params.update(exp["hyperparameter"])
        del exp["hyperparameter"]
    
    exp.update(params)
    agent = exp["model"]
    
    del exp["model"]
    
    # add global experiment parameters
    exp.update(global_exp_params)
    
    # convert dict to argparse object
    args = argparse.Namespace(**exp)
    
    if args.env == "default":
        env = agent.create_env(**default_env)
    
    # calculate the optimal reward possible for the environment
    # optimal_reward = args.max_steps / (env.columns-1)
    # args.optimal_reward = optimal_reward
    
    for i in range(number_of_repeats):
        agent.train(env, args)
        