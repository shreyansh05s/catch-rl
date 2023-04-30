import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse
from catch import Catch
import os

#MODELS
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


class ACagent:
    
    def __init__(self, input_size, output_size, hidden_size, gamma, lr, depth, trace_length, entropy_strength, device, use_baseline=False, use_bootstrapping=False):
        self.actor = Policy(input_size, output_size, hidden_size).to(device)
        self.critic = Value(input_size, hidden_size).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.device = device
        self.use_baseline = use_baseline
        self.use_bootstrapping = use_bootstrapping
        self.entropy_weight = entropy_strength
        self.trace_length = trace_length
        self.depth = depth

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.actor(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def sample_trace(self, env):
        done = False
        state = env.reset()
        trace = []
        reward = 0
        while not done:
            action, log_prob = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trace.append((state, action, reward, log_prob))
            reward += reward
            state = next_state
        return trace, reward

    def update(self, trace):
        actor_loss = 0
        critic_loss = 0
        R = 0
        n = self.depth
        for i in range(len(trace)-1):
            V = self.critic.forward(torch.from_numpy(trace[i+n][0]).float().unsqueeze(0).to(self.device))
            Q_n = sum([trace[i + k][2] for k in range(n)]) + V
            v_pred = self.critic.forward(trace[i][0], self.device)

            if self.use_baseline:
                advantage = Q_n - v_pred
            else:
                advantage = Q_n

            actor_loss = actor_loss - trace[i][3] * advantage.detach()
            actor_loss -= self.entropy_weight * trace[i][3].entropy()
            critic_loss = critic_loss + nn.MSELoss()(v_pred, Q_n)

        #update the actor and critic
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

    def train(self, env, num_episodes):
        for i in range(num_episodes):
            trace, reward = self.sample_trace(env)
            self.update(trace)
            print("Episode: {}, Reward: {}".format(i, reward))

    def test(self, env, num_episodes):
        for i in range(num_episodes):
            done = False
            state = env.reset()
            reward = 0
            while not done:
                env.render()
                action, _ = self.select_action(state)
                state, r, done, _ = env.step(action)
                reward += r
            print("Episode: {}, Reward: {}".format(i, reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--env", default="Catch", type=str)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--max_episode_length", default=1000, type=int)
    parser.add_argument("--trace_length", default=100, type=int)
    parser.add_argument("--entropy_strength", default=0.01, type=float)
    parser.add_argument("--use_baseline", default=False, type=bool)
    parser.add_argument("--use_bootstrapping", default=False, type=bool)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--save_path", default="saved_models", type=str)
    parser.add_argument("--load_path", default="saved_models", type=str)
    parser.add_argument("--depth", default=10, type=int)
    args = parser.parse_args()

    env = Catch()
    agent = ACagent(env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2], env.action_space.n, args.hidden_size, args.gamma, args.lr, args.depth, args.trace_length, args.entropy_strength, args.device, args.use_baseline, args.use_bootstrapping)

    if args.mode == "train":
        agent.train(env, args.num_episodes)
    elif args.mode == "test":
        agent.test(env, args.num_episodes)
    else:
        raise NotImplementedError

