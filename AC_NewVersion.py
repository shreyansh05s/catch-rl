import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from catch import Catch
import argparse
import gym
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
import argparse
# import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        # use input of 7x7x2
        self.fc0 = nn.Flatten()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x


# Define the value network
class Value(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Value, self).__init__()
        self.fc0 = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_advantages(env, input_size, hidden_size, delta, use_baseline, use_bootstrapping, gamma):
    critic = Value(input_size, hidden_size).to(device)
    if use_baseline:
        delta = delta - torch.mean(delta)
    if use_bootstrapping:
        state = env.reset()
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        delta += gamma * critic(state).squeeze()
    return delta

def train(env, num_episodes, num_traces, lr, gamma, hidden_size, entropy_strength, wandb_project, use_entropy=True, use_bootstrapping=True, use_baseline=True):
    try:
        input_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    except:
        # for vector observations
        input_size = env.observation_space.shape[0]
    
    actor = Policy(input_size, hidden_size, env.action_space.n).to(device)
    critic = Value(input_size, hidden_size).to(device)

    optimizer_actor = optim.SGD(actor.parameters(), lr=0.001)
    optimizer_critic = optim.SGD(critic.parameters(), lr=0.001)

    for episode in range(num_episodes):
        policy_loss = torch.tensor([0], dtype=torch.float64, device=device) 
        value_loss = torch.tensor([0], dtype=torch.float64, device=device)
        total_reward = 0

        for trace in range(num_traces):
            state = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            log_probs = []
            

            while not done:
                trace = []
                current_reward = 0
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
                policy_probs = actor(state)
                action_dist = torch.distributions.Categorical(policy_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                next_state, reward, done, _ = env.step(action.item())

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                trace.append((state, action, reward, action_dist))


                state = next_state
                current_reward += reward

            total_reward = total_reward + current_reward

            for h in range(len(trace)-1):
                depth = min(depth, (len(trace) - 1 - h))
                values = Value.forward(trace[h + depth][0], device)
                qvalues = sum([trace[h + k][2] for k in range(depth)]) + values
                v_pred = Value.forward(trace[h][0], device)

                if not use_baseline:
                    policy_loss += qvalues.detach() * -trace[h][3].log_probs(trace[h][1])
                else:
                    policy_loss += (qvalues.detach() - v_pred.detach()) * -trace[h][3].log_probs(trace[h][1])
                    
                if use_entropy:
                    policy_loss -= entropy_strength * trace[h][3].entropy()

                value_loss += torch.square(qvalues.detach() - v_pred)
                loss = policy_loss + value_loss

            policy_loss /= num_traces
            value_loss /= num_traces
            total_reward /= num_traces

            # values = critic(torch.stack(states).to(device)).squeeze()
            # next_values = torch.cat((values[1:], torch.zeros(1).to(device)))
            # rewards = torch.FloatTensor(rewards).to(device)
            # delta = rewards + gamma * next_values - values
            # advantages = compute_advantages(env, input_size, hidden_size, delta, use_baseline, use_bootstrapping, gamma)

        # # Compute policy loss
        # log_probs = torch.stack(log_probs)
        # policy_loss = -torch.mean(log_probs * advantages) - entropy_strength * torch.mean(policy_probs * torch.log(policy_probs))

        # # Compute value loss
        # targets = rewards + gamma * next_values
        # value_loss = nn.MSELoss()(values, targets)

        # Compute total loss
        loss = policy_loss + value_loss

        # Update actor and critic networks
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        loss.backward()
        optimizer_actor.step()
        optimizer_critic.step() 

        print("Episode: {}, total reward: {}".format(episode, total_reward))
        # wandb.log({'total_reward': total_reward, 'episode': episode})
    return total_reward


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Train an Actor-Critic RL agent for the Catch environment')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the hidden layer')
    parser.add_argument('--num_episodes', type=int, default=5000, help='number of episodes to train')
    parser.add_argument('--entropy_strength', type=float, default=1.0, help='entropy strength')
    parser.add_argument('--num_traces', type=float, default=10, help='number of traces')
    parser.add_argument('--wandb_project', type=str, default='catch-rl', help='Wandb project name')
    args = parser.parse_args()

    # wandb.init(project=args.wandb_project, config=args)

    env = Catch(rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, observation_type='pixel', seed=None)
    train(env, args.num_episodes, args.num_traces, args.lr, args.gamma, args.hidden_size, args.entropy_strength, args.wandb_project)
    
    # wandb.finish()
 