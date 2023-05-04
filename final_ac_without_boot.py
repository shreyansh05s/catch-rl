import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from catch import Catch

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.85, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

observation_type = 'pixel'

env = Catch(rows=7, columns=7, speed=1.0, max_steps=250,
            max_misses=10, observation_type=observation_type, seed=None)
# env.reset(seed=args.seed)
# torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 2, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 3)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1) # Flatten the tensor
        x = torch.relu(self.fc1(x))
        

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    state = state.unsqueeze(0)
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)
    
    # add entropy to encourage exploration
    

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value, m.entropy().item()))

    # the action to take (left or right)
    return action.item()


def finish_episode(baseline=True, entropy_regularization=False, entropy_weight=0.01, normalize_returns=True, bootstrap=True):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    if not bootstrap:
        # calculate the true value using rewards returned from the environment
        # monte carlo approach for adding all rewards
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        
        # normalize the true values (this is not 100% necessary)
        # show experimentally that this helps to stabilize training
        if normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + eps)
            
        # implement without bootstrapping where bootstrap is the value of the last state
        #if bootstrap:
                

        for (log_prob, value, entropy), R in zip(saved_actions, returns):
            if baseline:
                advantage = R - value.item()
            else:
                advantage = R
            
            # add entropy to encourage exploration
            if entropy_regularization:
                log_prob += entropy * entropy_weight
            
            # calculate actor (policy) loss        
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            # value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            
            value_losses.append(F.mse_loss(value, torch.tensor([R])))


    else:
        # use bootstrapping to sample from n steps
        bootstrap_returns = []
        n = 50
        # create empty lists for each bootstrap return
        bootstrap_returns = [[] for i in range(len(model.rewards) - n)]
        
        for i in range(len(model.rewards) - n):
            # bootstrap_returns[i] = []
            for j in range(n-1):
                R += args.gamma ** j * model.rewards[i + j]
                bootstrap_returns[i].insert(0, R)
                
        # convert each bootstrap_returns to tensor
        bootstrap_returns = [torch.tensor(bootstrap_return) for bootstrap_return in bootstrap_returns]
        
        # normalize each bootstrap_returns
        if normalize_returns:
            bootstrap_returns = [(bootstrap_return - bootstrap_return.mean()) / (bootstrap_return.std() + eps) for bootstrap_return in bootstrap_returns]
        
        # calculate the loss for each bootstrap_returns
        for i in range(len(bootstrap_returns)):
            for (log_prob, value, entropy), R in zip(saved_actions, bootstrap_returns[i]):
                if baseline:
                    advantage = R - value.item()
                else:
                    advantage = R

                # add entropy to encourage exploration
                if entropy_regularization:
                    log_prob += entropy * entropy_weight

                # calculate actor (policy) loss        
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                # value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

                value_losses.append(F.mse_loss(value, torch.tensor([R])))


    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()
    
    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        done = False
        while not done:
            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            # finish_episode()
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()
        scheduler.step()
        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, lr: {:.5f}'.format(
                  i_episode, ep_reward, running_reward, scheduler.get_lr()[0]))


if __name__ == '__main__':
    main()
