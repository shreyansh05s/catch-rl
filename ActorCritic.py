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
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
optimizer = None
scheduler = None

# utilized for normalizing the returns
eps = np.finfo(np.float32).eps.item()

# create a tuple to store the information about our transitions
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])


class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, input_size=7, hidden_size1=16, hidden_size2=32, hidden_size3=128, actor_output=3, critic_output=1, observation_type="pixel"):
        super(ActorCritic, self).__init__()

        self.observation_type = observation_type

        if self.observation_type == "pixel":
            self.conv1 = nn.Conv2d(
                input_size, hidden_size1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(
                hidden_size1, hidden_size2, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(hidden_size2 * input_size * 2, hidden_size3)
        elif self.observation_type == "vector":
            self.fc1 = nn.Linear(input_size, hidden_size2)
            self.fc2 = nn.Linear(hidden_size2, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)

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
        if self.observation_type == "pixel":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.reshape(x.size(0), -1)  # Flatten the tensor
            x = torch.relu(self.fc1(x))

        elif self.observation_type == "vector":
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))

        # actor
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic
        state_values = self.value_head(x)

        return action_prob, state_values


def select_action(state):
    # state = torch.FloatTensor(state).unsqueeze(0)
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


def update(gamma, entropy_weight, baseline=False, entropy_regularization=False, normalize_returns=True, bootstrap=False, bootstrap_state=None, done=True):
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

    returns = torch.tensor(returns).to(device)

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
        value_losses.append(F.mse_loss(value, torch.tensor([R]).to(device)))

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


def add_padding(env, state):
    padding = abs(
        env.observation_space.shape[0] - env.observation_space.shape[1])

    if env.observation_space.shape[0] > env.observation_space.shape[1]:
        padding_tensor = torch.ones(
            padding, env.observation_space.shape[0], 2) * -1
        return torch.cat((state, padding_tensor.to(device)), dim=0)
    else:
        padding_tensor = torch.ones(
            env.observation_space.shape[1], padding, 2) * -1
        return torch.cat((state, padding_tensor.to(device)), dim=1)


def train(env, args):
    if args.experiment and args.wandb:
        wandb.init(project=args.wandb_project, config=args,
                   group=args.group, job_type=args.job_type)
        # add optimal reward as a reference to wandb graphs
        # wandb.run.tags = [f"Optimal Reward: {args.optimal_reward}"]

    global model, optimizer, scheduler

    # padding to be added to make the pixel observation square
    # in a scenario where the observation is not square
    padding_required = env.observation_space.shape[0] != env.observation_space.shape[1]

    # make input_size take value from the environment
    hidden_size = 16

    input_size = env.observation_space.shape[0]

    # and make hidden_size a variable parameter based on the input_size
    model = ActorCritic(input_size=input_size, hidden_size1=hidden_size, hidden_size2=32, hidden_size3=128,
                        actor_output=3, critic_output=1, observation_type=args.observation_type).to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize scheduler
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

    running_reward = 10
    # run infinitely many episodes
    for i_episode in range(args.num_episodes):

        # reset environment and episode reward
        if padding_required:
            state = add_padding(env, torch.FloatTensor(
                env.reset())).unsqueeze(0).to(device)
        else:
            state = torch.FloatTensor(env.reset()).unsqueeze(0).to(device)

        ep_reward = 0

        done = False
        while not done:

            # select action from policy
            action = select_action(state)

            # take the action
            next_state, reward, done, _ = env.step(action)
            if padding_required:
                next_state = add_padding(env, torch.FloatTensor(
                    next_state)).unsqueeze(0).to(device)
            else:
                next_state = torch.FloatTensor(
                    next_state).unsqueeze(0).to(device)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            if done and not args.bootstrap:
                update(args.gamma, args.entropy_weight, args.baseline, args.entropy_regularization,
                       args.normalize_returns, args.bootstrap)

            elif args.bootstrap and len(model.rewards) >= args.n_steps:

                # update the policy after every n steps
                bootstrap_state = next_state
                update(args.gamma, args.entropy_weight, args.baseline, args.entropy_regularization,
                       args.normalize_returns, args.bootstrap, bootstrap_state=bootstrap_state, done=done)

                # reset the rewards and action buffer
                del model.rewards[:]
                del model.saved_actions[:]

            state = next_state

        if args.normalize_graph:
            # convert the reward to a value between -1 and 1
            # with the min reward being args.min_reward and optimal reward being args.optimal_reward
            if ep_reward < 0:
                ep_reward = ep_reward / abs(args.min_reward)
            else:
                ep_reward = ep_reward / args.optimal_reward

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # update learning rate
        if args.lr_scheduler:
            scheduler.step()

        # log results on console
        if i_episode % args.log_interval == 0 and args.verbose:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, lr: {:.5f}'.format(
                  i_episode, ep_reward, running_reward, scheduler.get_lr()[0]))

        # log results on wandb
        if args.wandb:
            if args.lr_scheduler:
                wandb.log({'total_reward': ep_reward,
                           'avg_reward': running_reward, 'lr': scheduler.get_lr()[0]})
            else:
                wandb.log({'total_reward': ep_reward,
                           'avg_reward': running_reward, 'lr': args.lr})

    if args.experiment and args.wandb:
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
                        default=True, help='print training info')
    parser.add_argument('--observation_type', type=str,
                        default='pixel', help='pixel or vector')
    parser.add_argument('--num_episodes', type=int,
                        default=1000, help='number of episodes to train')
    parser.add_argument('--experiment', type=bool,
                        default=False, help='run experiment')
    parser.add_argument('--lr_decay', type=float,
                        default=0.99, help='learning rate decay')
    parser.add_argument('--lr_step_size', type=int,
                        default=50, help='learning rate step size')
    parser.add_argument('--job_type', type=str,
                        default='train', help='job type')
    parser.add_argument('--lr_scheduler', type=bool,
                        default=True, help='use learning rate schedule')
    parser.add_argument('--wandb', type=bool, default=False, help='use wandb')
    parser.add_argument('--env_rows', type=int, default=7, help='env rows')
    parser.add_argument('--env_columns', type=int,
                        default=7, help='env columns')
    parser.add_argument('--env_speed', type=float,
                        default=1.0, help='env speed')
    parser.add_argument('--env_max_steps', type=int,
                        default=250, help='env max steps')
    parser.add_argument('--env_max_misses', type=int,
                        default=10, help='env max misses')
    parser.add_argument('--normalize_graph', type=bool,
                        default=False, help='normalize graph')
    parser.add_argument('--cpu', type=bool,
                        default=False, help='use cpu')
    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')

    # initialize wandb for logging
    if args.wandb:
        wandb.init(project=args.wandb_project, config=args,
                   group=args.group, job_type=args.job_type)

    # set up environment
    env = create_env(rows=args.env_rows, columns=args.env_columns, speed=args.env_speed,
                     max_steps=args.env_max_steps, max_misses=args.env_max_misses, observation_type=args.observation_type)

    # run training
    train(env, args)

    if args.wandb:
        wandb.finish()
