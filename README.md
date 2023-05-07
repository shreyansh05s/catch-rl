# Policy Gradient Methods for Catch

## Setup

### Create Virtual Environment
```bash
python -m venv venv
```
### Activate Virtual Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```


### Wandb login
This project uses [wandb](https://wandb.ai/) for logging. To use wandb, you must first login to your account. To login, run the following command:
```bash
wandb login
```
#
## Running the Code

### Reinforce
This will run the baseline Reinforce algorithm with the default hyperparameters.
```bash
python Reinforce.py
```

### Actor-Critic
This will run the baseline Actor-Critic algorithm with the default hyperparameters.
```bash
python ActorCritic.py
```
hyperparameters can be changed by passing in the following arguments (all arguments are optional and this is a combined list of all arguments that can be passed in to the `ActorCritic.py` file or the `Reinforce.py` file)
- `--lr`: learning rate
- `--gamma`: discount factor
- `--num_episodes`: number of episodes to train for
- `--num_steps`: number of steps to train for
- `--seed`: random seed
- `--render`: whether to render the environment
- `--wandb`: whether to log to wandb
- `--wandb_project`: the name of the wandb project to log to
- `--n_steps`: number of steps to take before updating the policy
- `--bootstrap`: whether to use bootstrapping
- `--baseline`: whether to use a baseline
- `--entropy_regularization`: whether to use entropy regularization
- `--entropy_weight`: the weight of the entropy regularization
- `--normaliza_returns`: whether to normalize the returns
- `--group`: the name of the group to log to (wandb)
- `--job_type`: the type of job to run (wandb)
- `--number_of_repeats`: the number of times to repeat the experiment (wandb)
- `--lr_scheduler`: whether to use a learning rate scheduler (step decay)
- `--lr_decay`: the decay rate of the learning rate scheduler
- `--lr_step_size`: the step size of the learning rate scheduler
- `--env_speed`: the speed of the environment
- `--env_size`: the size of the environment
- `--env_max_steps`: the maximum number of steps the environment can take
- `--env_max_misses`: the maximum number of misses the environment can take
- `--normalize_graphs`: whether to normalize the graphs for comparision of varying rewards in different environments
- `--cpu`: whether to use the cpu

#
## Running Experiments

All experiments are run from the `experiments.py` file. The `experiments.py` file takes in three arguments:
- `--experiment`: the name of the experiment to run
- `--job_type`: the type of job to run (e.g. `train`, `test`, `train_and_test`)
- `--number_of_repeats`: the number of times to repeat the experiment

All experiments are defined in the `experiments.json` file. The `experiments.json` file contains a list of experiments. Each experiment has the following fields:
- `group`: the name of the group the experiment belongs to
- `job_type`: the type of job to run (e.g. `A2C`, `Reinforce`, `A2C-Baseline`, etc.)
- `number_of_repeats`: the number of times to repeat the experiment
- `model`: the name of the model to use (e.g. `ActorCritic`, `Reinforce`)
- `observation_type`: the type of observation to use (e.g. `pixel`, `vector`)
- `hyperparameters`: the hyperparameters to use for the experiment (all hyperparameters that can be passed into the `ActorCritic.py` or `Reinforce.py` files can be passed in here)


### Runs all experiments
```bash
python experiments.py
```

### Runs a specific experiment
```bash
python experiments.py --experiment <experiment_name>
```

### Runs a specific experiment with a specific job_type
```bash
python experiments.py --experiment <experiment_name> --job_type <job_type>
```

### Runs a specific experiment with a specific job_type and a number_of_repeats
```bash
python experiments.py --experiment <experiment_name> --job_type <job_type> --number_of_repeats <number_of_repeats>
```

#
## Viewing Results

### Wandb

All results are logged to [wandb](https://wandb.ai/). To view the results, go to the [wandb](https://wandb.ai/) website and login. Then, go to the project you want to view and select the group you want to view. You can then view the results of each experiment by selecting the corresponding group.

We would have uploaded the image of the wandb dashboard of our experiments form a visual representation on Brightspace but it does not support uploading images.