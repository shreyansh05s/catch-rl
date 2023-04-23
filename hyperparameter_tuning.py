import torch.optim as optim
from DQN import DQNAgent
import itertools

import gym
import numpy as np
from joblib import parallel_backend
import time


def grid_search(params):
    env = gym.make('CartPole-v1')

    best_reward = float('-inf')
    best_params = None
    with parallel_backend('multiprocessing', n_jobs=-1):
        start_time = time.time()
        for param_comb in itertools.product(*params.values()):    
            param_dict = dict(zip(params.keys(), param_comb))

            agent = DQNAgent(state_dim=4, action_dim=2,num_episodes=1000,hidden_dim=64,policy='softmax',tuning=True,plot=False,model='DQN', **param_dict)

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

    # 'state_dim': [2, 4, 6],
    # 'hidden_dim': [64],
    #'num_episodes': [500],
    # 'action_dim': [1, 2, 3],
    'lr': [0.001,0.01],
    'gamma': [0.9, 0.7,0.5],
    'batch_size': [32, 64],
    # 'target_update': [50, 100, 200],
    #'policy': ['egreedy','annealing_egreedy'],
    #'policy': [ 'softmax'],
    #'policy': [ 'annealing_egreedy'],
    # 'model': ['DQN', 'DoubleDQN', 'DuelingDQN'],
    # 'epsilon': [0.9, 0.5, 0.2],
    # 'buffer_size': [10000, 30000,100000],
    #'max_steps': [100, 500, 1000, 2000],
    # 'eps_start': [1.0, 0.5, 0.2, 0.9],
    #'eps_end': [0.01, 0.1],
    # 'eps_decay': [0.5, 0.7, 0.9],
    'temp': [0.2, 0.5, 0.8],
    
}

grid_search(params)
