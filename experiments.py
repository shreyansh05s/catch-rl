import wandb
import Reinforce
import ActorCritic
import argparse
from copy import deepcopy
import json
import math

# models
models = {
    "ActorCritic": ActorCritic,
    "Reinforce": Reinforce
}

# default hyperparameters
hyperparameter_defaults = {
    "lr": 1e-2,
    "gamma": 0.85,
    "n_steps": 100,
    "entropy_weight": 0.01,
    "entropy_regularization": True,
    "bootstrap": False,
    "baseline": False,
    "normalize_returns": True,
    "lr_decay": 0.99,
    "lr_step_size": 50,
    "lr_scheduler": True
}

# define different experiments in a list
global_exp_params = {
    "wandb_project": "catch-rl-experiments",
    "render": False,
    "num_episodes": 1000,
    "log_interval": 10,
    "experiment": True,
    "verbose": False,
    "wandb": True,
    "normalize_graph": False
}

default_env = {
    "rows": 7,
    "columns": 7,
    "speed": 1.0,
    "max_steps": 250,
    "max_misses": 10,
    "observation_type": "pixel",
}


def run_experiment(experiment, job_type=None, number_of_repeats=10):
    # run each experiment 10 times
    if job_type is not None:
        experiment = [exp for exp in experiment if exp["job_type"] == job_type]
    
    for exp in experiment:
        # run each experiment 10 times
        exp["model"] = models[exp["model"]]
        params = deepcopy(hyperparameter_defaults)
        if "hyperparameter" in exp:
            params.update(exp["hyperparameter"])
            del exp["hyperparameter"]

        exp.update(params)
        agent = exp["model"]

        del exp["model"]

        # add global experiment parameters which are not present in the hyperparameters
        exp.update({k:v for k, v in global_exp_params.items() if k not in exp})

        # convert dict to argparse object
        args = argparse.Namespace(**exp)


        env_params = deepcopy(default_env)

        # update only the parameters that are present in the default env
        env_params.update({k: v for k, v in exp.items() if k in default_env})
        
        # create environment
        env = agent.create_env(**env_params)
        
        if args.normalize_graph:
            args.optimal_reward = math.floor(env_params["max_steps"] / env_params["rows"])
            args.min_reward = env_params["max_misses"]

        for i in range(number_of_repeats):
            agent.train(env, args)



if __name__ == "__main__":
    run_all = False
    job_type = None
    ############
    # also add team name as an argument to run_experiment
    ############
    
    parser = argparse.ArgumentParser(
        description='Experiments for Catch environment')
    parser.add_argument('--experiment', type=str, default=None, metavar='N',
                        help='experiment to run (default: would run all experiments)')
    parser.add_argument('--job_type', type=str, default=None, metavar='N',
                        help='job type to run (default: would run all jobs)')
    parser.add_argument('--num_of_repeats', type=int, default=10, metavar='N',
                        help='number of times to repeat each experiment (default: 10)')
    args = parser.parse_args()

    if args.experiment is None:
        # run all experiments
        run_all = True

    # load experiments from json file
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    if "num_of_repeats" in experiment:
        args.num_of_repeats = experiment["num_of_repeats"]
    
    if run_all:
        for key, experiment in experiments.items():
            print("Running experiment: {}".format(key))
            run_experiment(experiment, args.num_of_repeats)

    else:
        print("Running experiment: {}".format(args.experiment))
        run_experiment(experiments[args.experiment], args.job_type, args.num_of_repeats)
