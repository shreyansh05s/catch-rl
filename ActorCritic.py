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
        self.conv1 = nn.Conv2d(input_size, hidden_size1,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size1, hidden_size2,
                               kernel_size=3, padding=1)
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
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))

        # actor: choses action to take from state s_t by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        return action_prob, state_values


model = None
optimizer = None
scheduler = None
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
    model.saved_actions.append(SavedAction(
        m.log_prob(action), state_value, m.entropy().item()))

    # the action to take (left, right, or stay idle)
    return action.item()


def update(gamma, entropy_weight, baseline=False, entropy_regularization=False, normalize_returns=True, bootstrap=False, bootstrap_state=None, done=False):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # use bootstrapping to estimate the n step return
    for i, r in enumerate(model.rewards[::-1]):
        # calculate the discounted value
        if bootstrap and i == 0 and not done:
            R = r + (gamma * model(bootstrap_state)[1].item())
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


def train(env, args):
    if args.experiment:
        wandb.init(project=args.wandb_project, config=args,
                   group=args.group, job_type=args.job_type)
        # add optimal reward as a reference to wandb graphs
        # wandb.run.tags = [f"Optimal Reward: {args.optimal_reward}"]
    
    global model, optimizer, scheduler
    
    model = Policy(input_size=7, hidden_size1=16, hidden_size2=32,
               hidden_size3=128, actor_output=3, critic_output=1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

    running_reward = 10
    # run infinitely many episodes
    for i_episode in range(args.num_episodes):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        done = False
        while not done and i_episode <= args.num_episodes:

            # select action from policy
            action = select_action(state)

            # take the action
            next_state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            if done and not args.bootstrap:
                update(args.gamma, args.entropy_weight, args.baseline, args.entropy_regularization,
                       args.normalize_returns, args.bootstrap, bootstrap_state=None, done=True)

            elif args.bootstrap and len(model.rewards) >= args.n_steps:

                # update the policy after every n steps
                bootstrap_state = torch.FloatTensor(next_state).unsqueeze(0)
                update(args.gamma, args.entropy_weight, args.baseline, args.entropy_regularization,
                       args.normalize_returns, args.bootstrap, bootstrap_state=bootstrap_state, done=done)

                # reset the rewards and action buffer
                del model.rewards[:]
                del model.saved_actions[:]

            state = next_state

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # update learning rate
        if args.lr_scheduler:
            scheduler.step()

        # log results
        if i_episode % args.log_interval == 0 and args.verbose:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, lr: {:.5f}'.format(
                  i_episode, ep_reward, running_reward, scheduler.get_lr()[0]))

        wandb.log({'total_reward': ep_reward,
                  'avg_reward': running_reward, 'lr': scheduler.get_lr()[0]})
    if args.experiment:
        wandb.finish()
    return running_reward


def create_env(rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, observation_type='pixel'):
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=None)
    return env


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch actor-critic example')
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
    parser.add_argument('--wandb_project', type=str,
                        default='catch-rl', help='Wandb project name')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--entropy_weight', type=float,
                        default=0.01, help='strength of entropy regularization')
    parser.add_argument('--baseline', type=bool, default=False,
                        help='use of baseline subtraction')
    parser.add_argument('--bootstrap', type=bool,
                        default=False, help='use of bootstrapping')
    parser.add_argument('--entropy_regularization', type=bool,
                        default=False, help='use of entropy regularization')
    parser.add_argument('--normalize_returns', type=bool,
                        default=True, help='normalise returns')
    parser.add_argument('--group', type=str, default='test',
                        help='Wandb group name')
    parser.add_argument('--verbose', type=bool,
                        default=False, help='print training info')
    parser.add_argument('--observation_type', type=str,
                        default='pixel', help='pixel or vector')
    parser.add_argument('--num_episodes', type=int,
                        default=1000, help='number of episodes to train')
    parser.add_argument('--experiment', type=bool, default=False, help='run experiment')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay')
    parser.add_argument('--lr_step_size', type=int, default=50, help='learning rate step size')
    parser.add_argument('--job_type', type=str, default='train', help='job type')
    parser.add_argument('--lr_scheduler', type=bool, default=True, help='use learning rate schedule')
    args = parser.parse_args()

    num_episodes = args.num_episodes
    observation_type = args.observation_type

    wandb.init(project=args.wandb_project, config=args,
               group=args.group, job_type=args.job_type)

    # set up environment
    env = create_env()

    train(env, args)

    wandb.finish()
