import wandb
import Reinforce
import ActorCritic
import argparse
from copy import deepcopy
import json

# models
models = {
    "ActorCritic": ActorCritic,
    "Reinforce": Reinforce
}

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
    "num_episodes": 1000,
    "log_interval": 10,
    "experiment": True,
    "verbose": False,
}

default_env = {
    "rows": 7,
    "columns": 7,
    "speed": 1.0,
    "max_steps": 250,
    "max_misses": 10,
    "observation_type": "pixel",
}

number_of_repeats = 10


def run_experiment(experiment):
    # run each experiment 10 times
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



if __name__ == "__main__":
    run_all = False
    
    parser = argparse.ArgumentParser(
        description='Experiments for Catch environment')
    parser.add_argument('--experiment', type=str, default=None, metavar='N',
                        help='experiment to run (default: would run all experiments)')
    args = parser.parse_args()

    if args.experiment is None:
        # run all experiments
        run_all = True

    # load experiments from json file
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    if run_all:
        for key, experiment in experiments.items():
            print("Running experiment: {}".format(key))
            # print(experiment)
            run_experiment(experiment)

    else:
        print("Running experiment: {}".format(args.experiment))
        run_experiment(experiments[args.experiment])
