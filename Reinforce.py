import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from catch import Catch
import wandb
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define the policy network
# class Policy_network(nn.Module):
#     def __init__(self, input_size,hidden_size, output_size):
#         super(Policy_network, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
        
       
        
#     def forward(self, x):
#         x = self.flatten(x) # Flatten the input tensor
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Define the policy network
class Policy_network(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super(Policy_network, self).__init__()
        #create convolutional layers with  input[1, 7, 7, 2]
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
        return x

# Define the REINFORCE agent
class Reinforce:
    def __init__(self, input_size,hidden_size, output_size, lr=0.001, gamma=0.99,entropy_reg = True, beta = 0.001):
        self.policy = Policy_network(input_size,hidden_size, output_size).to(device) # Create the policy network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr) # Create the optimizer
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        self.entropy_reg = entropy_reg
        self.beta = beta
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)   # Convert the state to a tensor
        action_probs = torch.softmax(self.policy(state), dim=1).to(device)  # Get the action probabilities
        m = Categorical(action_probs)  # Create a categorical distribution over the action probabilities
        action = m.sample() # Sample an action from the distribution
        self.saved_log_probs.append(m.log_prob(action)) # Save the log probability of the action
        return action.item()
    
    def update_policy(self):
        R = 0 
        policy_loss = [] 
        entropy_loss = [] 
        returns = []
        # loop through the rewards in reverse order
        for r in self.rewards[::-1]: 
            R = r + self.gamma * R # Calculate the discounted return
            returns.insert(0, R) # Insert the discounted return at the beginning of the list
        returns = torch.tensor(returns)  
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Normalize the returns

        # loop through the saved log probabilities and the returns
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R) # Calculate the policy loss
            entropy_loss.append(-(torch.exp(log_prob) * log_prob)) # Calculate the entropy loss
        self.optimizer.zero_grad() # Reset the gradients 
        policy_loss = torch.cat(policy_loss).to(device).sum() # Calculate the sum of the policy loss
        entropy_loss = torch.cat(entropy_loss).to(device).sum() # Calculate the sum of the entropy loss
        if self.entropy_reg: # If entropy regularization is used
            loss = policy_loss + self.beta * entropy_loss # Calculate the loss as a weighted sum
        else:
            loss = policy_loss
        loss.backward() # Calculate the gradients
        self.optimizer.step() # Update the policy network
        self.saved_log_probs = [] # Reset the saved log probabilities
        self.rewards = [] # Reset the rewards

        wandb.log({'policy_loss': policy_loss.item(), 'entropy_loss': entropy_loss.item(), 'loss': loss.item()})
    
    # Get the action with the highest probability from the policy network during testing
    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        action_probs = torch.softmax(self.policy(state), dim=1).to(device)
        action = action_probs.argmax(dim=1).to(device)
        return action.item()

    
# Define the REINFORCE training function
def train(env, agent, num_episodes=1000, max_steps=250, print_interval=100):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        for step in range(max_steps):
            # env.render() 
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.rewards.append(reward)
            state = next_state
            if done:
                break

        agent.update_policy()
        total_rewards.append(total_reward)
        if episode % print_interval == 0:
            print("Episode {}\tTotal Reward: {}".format(episode, total_reward))

        #Log the episode reward, loss ,TD error, and learning rate
        wandb.log({'total_reward': total_reward })
        
    return total_rewards

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Train an Reinforce agent for the Catch environment')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='discount factor')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the hidden layer')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to train')
    parser.add_argument('--beta', type=int, default=0.01, help='the strength of the entropy regularization term in the loss.')
    parser.add_argument('--entropy_reg', type=bool, default=False, help='add entropy regularization')
    parser.add_argument('--wandb_project', type=str, default='Reinforce', help='Wandb project name')
    args = parser.parse_args()

    

    # comment out to enable wandb logging
    wandb.init(mode='disabled')
     

    # Initialize Wandb logging inside team project "leiden-rl"
    # add team
    #wandb.init(project=args.wandb_project, config=args)    

    # Set boolean parameter
    wandb.config.entropy_reg = False

    # Log the hyperparameter values
    #wandb.log({'lr': args.lr, 'gamma': args.gamma, 'hidden_size': args.hidden_size, 'num_episodes': args.num_episodes, 'beta': args.beta, 'entropy_reg': args.entropy_reg})

    # Create an instance of the customizable Catch environment
    env = Catch(rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, observation_type='pixel', seed=None)

    # Create an instance of the REINFORCE agent
    input_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    output_size = env.action_space.n
    # hidden_size = 128


    agent = Reinforce(input_size,args.hidden_size,output_size,args.lr,args.gamma,args.entropy_reg,args.beta)

    # Train the agent
    num_episodes = 1000
    max_steps = 250
    print_interval = 100
    total_rewards = train(env, agent,args.num_episodes, max_steps, print_interval)

    

    print("Average rewards: {}".format(np.mean(total_rewards)))

    wandb.log({'Average rewards': np.mean(total_rewards)})
    wandb.finish()

