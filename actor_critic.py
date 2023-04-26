import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from catch import Catch
import torch.nn.functional as F
from torch.distributions import Categorical
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

def train_actor_critic(env, num_episodes, lr, gamma, hidden_size, wandb_project, use_bootstrapping=False, use_baseline=False):
    # Initialize the networks, optimizer and loss function
    input_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    
    policy_net = Policy(input_size, hidden_size, env.action_space.n).to(device)
    value_net = Value(input_size, hidden_size).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    
    scheduler_policy = optim.lr_scheduler.ExponentialLR(optimizer_policy, gamma)
    scheduler_value = optim.lr_scheduler.ExponentialLR(optimizer_value, gamma)
    
    mse_loss = nn.MSELoss()

    # Train the agent
    for i_episode in range(num_episodes):
        # Reset the environment and get the initial state
        state = torch.FloatTensor(env.reset()).to(device).unsqueeze(0)
        # print(state.shape)
        done = False
        total_reward = 0

        while not done:
            # Select an action using the policy network
            action_probs = policy_net(state)
            action = torch.distributions.Categorical(action_probs).sample()

            # Take the action in the environment and get the next state and reward
            next_state, reward, done, info = env.step(action.item())
            next_state = torch.FloatTensor(next_state).to(device).unsqueeze(0)

            # Calculate the TD error
            value = value_net(state).item()
            next_value = value_net(next_state).item()
            td_error = reward + gamma * next_value - value

            # Update the value network
            optimizer_value.zero_grad()
            if use_bootstrapping:
                # Use bootstrapping to calculate the target for the value network
                target_value = reward + gamma * next_value * (1 - done)
            else:
                # Use Monte Carlo returns as the target for the value network
                target_value = reward if done else reward + gamma * value_net(next_state).item()
            value_loss = mse_loss(torch.FloatTensor([target_value]).to(device), value_net(state))
            value_loss.backward()
            optimizer_value.step()

            # Update the policy network
            optimizer_policy.zero_grad()
            if use_baseline:
                # Calculate the advantage function using the value function as a baseline
                advantage = td_error - value
            else:
                # Calculate the advantage function without subtracting the value function
                advantage = td_error
            policy_loss = -torch.log(action_probs[action]) * advantage
            policy_loss.backward()
            optimizer_policy.step()

            # Update the state and total reward
            state = next_state
            total_reward += reward

        # take a step in the learning rate scheduler every epoch
        scheduler_value.step()
        scheduler_policy.step()
        
        lr = scheduler_value.get_last_lr()[0]

        # Log the episode reward, loss ,TD error, and learning rate
        # wandb.log({'total_reward': total_reward, 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'td_error': td_error, 'lr': lr})

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Train an Actor-Critic RL agent for the Catch environment')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the hidden layer')
    parser.add_argument('--num_episodes', type=int, default=5000, help='number of episodes to train')
    parser.add_argument('--wandb_project', type=str, default='catch-rl', help='Wandb project name')
    args = parser.parse_args()
    
    # Initialize Wandb logging inside team project "leiden-rl"
    # add team
    # wandb.init(project=args.wandb_project, config=args)
    
    # Create an instance of the customizable Catch environment
    env = Catch(rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, observation_type='pixel', seed=None)
    
    train_actor_critic(env, args.num_episodes, args.lr, args.gamma, args.hidden_size, args.wandb_project)
    
    # wandb.finish()
