import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from catch import Catch

# Define the policy network
class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize environment
env = Catch()

# Set hyperparameters
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 64
learning_rate = 0.001
gamma = 0.99

# Initialize policy network and optimizer
policy = Policy(input_size, output_size, hidden_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()  # Reset environment and get initial state
    rewards = []
    log_probs = []
    
    while True:
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        
        # Get action probabilities from policy network
        action_probs = torch.softmax(policy(state_tensor), dim=-1)
        
        # Choose action based on action probabilities
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob)
        
        # Take action in the environment
        next_state, reward, done, _ = env.step(action.item())
        rewards.append(reward)
        
        if done:
            break
        
        state = next_state
        
    # Calculate discounted rewards
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
        
    # Convert discounted rewards to tensor
    discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
    
    # Normalize discounted rewards
    discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean()) / (discounted_rewards_tensor.std() + 1e-9)
    
    # Calculate loss
    loss = torch.dot(torch.stack(log_probs).sum(),  -discounted_rewards_tensor.detach())
    
    # Update policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print episode information
    print('Episode [{}/{}], Total reward: {}'.format(episode+1, num_episodes, sum(rewards)))
