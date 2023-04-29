import torch
import torch.nn as nn
import torch.optim as optim
import gym
import wandb
import argparse
from catch import Catch
class PolicyConv(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyConv, self).__init__()
        
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1) # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
    
class ValueConv(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueConv, self).__init__()
        
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1) # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Define the policy network
class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Define the value network
class Value(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_actor_critic(env, num_episodes, lr, gamma, hidden_size, wandb_project, use_bootstrapping=False, use_baseline=False, lr_step_size=10):
    # Initialize the networks, optimizer and loss function
    policy_net = PolicyConv(env.observation_space.shape[0], hidden_size, env.action_space.n)
    value_net = ValueConv(env.observation_space.shape[0], hidden_size)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    scheduler_policy = optim.lr_scheduler.StepLR(optimizer_policy, step_size=lr_step_size, gamma=gamma)
    scheduler_value = optim.lr_scheduler.StepLR(optimizer_value, step_size=lr_step_size, gamma=gamma)
    # scheduler_policy = optim.lr_scheduler.ExponentialLR(optimizer_policy, gamma=0.99)
    # scheduler_value = optim.lr_scheduler.ExponentialLR(optimizer_value, gamma=0.99)
    mse_loss = nn.MSELoss()

    # Initialize Wandb logging
    # wandb.init(project=wandb_project)

    # Train the agent
    for i_episode in range(num_episodes):
        # Reset the environment and get the initial state
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            # Select an action using the policy network
            action_probs = policy_net(state)
            action = torch.distributions.Categorical(action_probs).sample()

            # Take the action in the environment and get the next state and reward
            next_state, reward, done, info = env.step(action.item())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
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
                target_value = reward if done else reward + gamma * value_net(torch.FloatTensor(next_state)).item()
            
            # fix shape of target_value to match value
                
            value_loss = mse_loss(torch.FloatTensor([target_value]), value_net(torch.FloatTensor(state)))
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
            action_probs = action_probs.squeeze(0)
            policy_loss = -torch.log(action_probs[action]) * advantage
            policy_loss.backward()
            optimizer_policy.step()

            # Update the state and total reward
            state = next_state
            total_reward += reward
            
        scheduler_policy.step()
        scheduler_value.step()

        # Log the episode reward
        wandb.log({'total_reward': total_reward, 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'td_error': td_error, 'lr': scheduler_policy.get_last_lr()[0]})
        # print('Episode {}\tTotal reward: {:.2f} learning rate: {:.2e}'.format(i_episode, total_reward, scheduler_policy.get_last_lr()[0]))

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Train an Actor-Critic RL agent for the Catch environment')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.85, help='discount factor')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the hidden layer')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to train')
    parser.add_argument('--wandb_project', type=str, default='catch-rl', help='Wandb project name')
    parser.add_argument('--use_bootstrapping', action='store_true', help='use bootstrapping instead of Monte Carlo returns')
    parser.add_argument('--use_baseline', action='store_true', help='use the value function as a baseline')
    parser.add_argument('--lr_step_size', type=int, default=10, help='number of episodes before decreasing learning rate')
    args = parser.parse_args()
    
    # Initialize Wandb logging inside team project "leiden-rl"
    # add team
    wandb.init(project=args.wandb_project, config=args)
    
    # Create an instance of the customizable Catch environment
    env = Catch(rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, observation_type='pixel', seed=None)
    
    train_actor_critic(env, args.num_episodes, args.lr, args.gamma, args.hidden_size, args.wandb_project)
    
    wandb.finish()
