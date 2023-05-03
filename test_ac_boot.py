import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from catch import Catch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env.reset(seed=seed)
# torch.manual_seed(seed)

# create a tuple to store the information about our transitions
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, input_size=7, hidden_size1=16, hidden_size2=32, hidden_size3=128, actor_output=3, critic_output=1):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size1, hidden_size2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_size2 * input_size * 2, hidden_size3)

        # actor's layer
        self.action_head = nn.Linear(hidden_size3, actor_output)

        # critic's layer
        self.value_head = nn.Linear(hidden_size3, critic_output)

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

model = Policy(input_size=7, hidden_size1=16, hidden_size2=32, hidden_size3=128, actor_output=3, critic_output=1)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value, m.entropy().item()))

    # the action to take (left, right, or stay idle)
    return action.item()


def update(gamma, entropy_weight, baseline=True, entropy_regularization=False, normalize_returns=True, bootstrap=False, bootstrap_state=None, done=False):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # use or not of bootstraping to estimate the n step return
    if bootstrap:
        for i, r in enumerate(model.rewards[::-1]):
            if i == 0:
                R = r + (gamma * model(bootstrap_state)[1].item())
            
            R = r + gamma * R
            returns.insert(0, R)
    else:
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.insert(0, R)

    returns = torch.tensor(returns)
    
    # normalize the returns to stabilize training
    if normalize_returns:
        returns = (returns - returns.mean()) / (returns.std() + eps)
        

    for (log_prob, value, entropy), R in zip(saved_actions, returns):

        # calculate advantage using or not baseline subtraction
        if baseline:
            advantage = R - value.item()

        else:
            advantage = R
        
        # add entropy to encourage exploration multiplied by a weight factor to control its strength
        if entropy_regularization:
            log_prob += entropy * entropy_weight 
        
        # calculate policy loss        
        policy_losses.append(-log_prob * advantage)

        # calculate value loss 
        value_losses.append(F.mse_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backpropagation
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.85, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--n_steps', type=int, default=40, metavar='N',
                        help='number of steps to train (default: 1000)')
    parser.add_argument('--wandb_project', type=str, default='catch-rl', help='Wandb project name')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='strength of entropy regularization')
    parser.add_argument('--baseline', type=bool, default=True, help='use of baseline subtraction')
    parser.add_argument('--bootstrap', type=bool, default=True, help='use of bootstrapping')
    parser.add_argument('--entropy_regularization', type=bool, default=True, help='use of entropy regularization')
    parser.add_argument('--normalize_returns', type=bool, default=True, help='normalise returns')
    args = parser.parse_args()

    
    running_reward = 10
    num_episodes = 1000
    observation_type = 'pixel'

    wandb.init(mode='disabled')
    # wandb.init(project=args.wandb_project, config=args)

    # set up environment
    env = Catch(rows=7, columns=7, speed=1.0, max_steps=250,
            max_misses=10, observation_type=observation_type, seed=None)


    # run infinitely many episodes
    for i_episode in range(num_episodes):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        done = False
        while not done and i_episode <= num_episodes:

            # select action from policy
            action = select_action(state)

            # take the action
            next_state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            if done and not args.bootstrap:
                update(args.gamma, args.entropy_weight, args.baseline, args.entropy_regularization, args.normalize_returns, args.bootstrap, bootstrap_state=None, done=True)

            elif args.bootstrap and len(model.rewards) >= args.n_steps:
                
                # update the policy after every n steps
                bootstrap_state = torch.FloatTensor(next_state).unsqueeze(0)
                update(args.gamma, args.entropy_weight, args.baseline, args.entropy_regularization, args.normalize_returns, args.bootstrap, bootstrap_state=bootstrap_state, done=False)
                
                # reset the rewards and action buffer to have only the last n steps
                # model.rewards = model.rewards[-(n_steps-1):]
                # model.saved_actions = model.saved_actions[-(n_steps-1):]
                del model.rewards[:]
                del model.saved_actions[:]
            
            state = next_state

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # update learning rate         
        scheduler.step()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, lr: {:.5f}'.format(
                  i_episode, ep_reward, running_reward, scheduler.get_lr()[0]))
            wandb.log({'Episode Reward': ep_reward, 'Average Reward': running_reward, 'Learning Rate': scheduler.get_lr()[0]})

       
    wandb.finish()

if __name__ == '__main__':
    main()

