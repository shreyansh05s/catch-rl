import torch.optim as optim
import itertools
import numpy as np
from joblib import parallel_backend
import time

from catch import Catch
from Reinforce import REINFORCEAgent


def grid_search(params):
    env = Catch()

    # Create an instance of the REINFORCE agent
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 128
    

    best_reward = float('-inf')
    best_params = None
    with parallel_backend('multiprocessing', n_jobs=-1):
        start_time = time.time()
        for param_comb in itertools.product(*params.values()):    
            param_dict = dict(zip(params.keys(), param_comb))

            agent = REINFORCEAgent(input_size, output_size,**param_dict)

            episode_rewards = agent.train(env)
            avg_reward = np.mean(episode_rewards)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_params = param_dict

            print("Average reward for this combination: ", avg_reward)
               
    end_time = time.time()
    print("Time taken: {} minutes".format((end_time - start_time)/60))
    print("Best hyperparameters found {} with average reward: {} ".format(best_params,best_reward))


params = {
#we choose the parameters we want to tune

    'hidden_size': [64, 128, 256],
    #'num_episodes': [500],
    'lr': [0.001,0.01],
    'gamma': [0.9, 0.7,0.5],
    
}

grid_search(params)
