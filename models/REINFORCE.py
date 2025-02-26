import os
from typing import Literal
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical       
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from module import RL
# import imageio                                  # imageio 是一个用于读写图像和视频的 Python 库

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class REINFORCE(nn.Module):
    def __init__(self, env, epochs=500, gamma=0.99, h_size=32, lr=1e-3, reward_threshold=None, baseline=0):       
        super().__init__()
        self.env = env
        self.action_num = sum(self.env.action_space.shape) if type(env.action_space) == gym.spaces.box.Box else self.env.action_space.n
        self.state_num = sum(self.env.observation_space.shape) if type(env.observation_space) == gym.spaces.box.Box else self.env.observation_space.n
        self.epochs = epochs
        self.gamma = gamma
        self.max_steps_per_eos = self.env.spec.max_episode_steps
        self.h_size = h_size
        self.fc1 = nn.Linear(self.state_num, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, self.action_num)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.reward_threshold = self.env.spec.reward_threshold  if reward_threshold is None else reward_threshold
        self.baseline = baseline
        self.epoch_rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))                      
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)  # The input of torch is a batch (2d), so the output is 2d (batch_size, action_num)，that's why softmax on the dim1.
        return x                                     
    
    def act(self, state, mode:Literal["train", "evaluate"] = "train"):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu().squeeze()
        if mode == "evaluate":
            action = probs.argmax()
            return action.item(), probs.log()[action.item()]
        else:
            prob_dist = Categorical(probs)
            action = prob_dist.sample()                                   
            return action.item(), prob_dist.log_prob(action)

    def train(self):
        self.epoch_rewards = []
        print(f"Env: {self.env.spec.id}\nState_space: {self.env.observation_space}\nAction_space: {self.env.action_space}")

        for epoch in range(self.epochs):
            log_probs = []
            rewards = []
            policy_reward = 0
            s = self.env.reset(seed=42)[0]

            while True:
                a, log_prob = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                log_probs.append(log_prob) 
                rewards.append(reward)                              
                if terminated or truncated:
                    break

            self.epoch_rewards.append(sum(rewards))
            steps = len(rewards)
            returns = deque()
               
            for t in range(steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)   
                returns.appendleft(self.gamma*disc_return_t + rewards[t]   )  # Dynamic Programming

            for log_prob, disc_return in zip(log_probs, returns):
                policy_reward += (-log_prob * (disc_return - self.baseline)).sum()  # log_prob * disc_return，but the default gradient descent direction is "-"，Adding "-" means gradient rise.

            policy_reward.to(device)                                              
            self.optimizer.zero_grad()
            policy_reward.backward()
            self.optimizer.step()

            avg_10_reward = sum(self.epoch_rewards[-10:])/10
            optimal_reward = self._evaluate()
            if epoch % 10 == 0:       
                print(f"Episode: {epoch}\tAverage reward: {avg_10_reward:.3f}\tOptimal reward: {optimal_reward}\tPolicy step_return: {policy_reward.item()/steps:.3f}")
            if avg_10_reward >= self.reward_threshold and optimal_reward >= self.reward_threshold:
                print(f"Converged at epoch: {epoch}, final optimal reward: {optimal_reward}")
                break

    def _evaluate(self):
        s = self.env.reset(seed=42)[0]
        rewards = 0
        while True:
            a, _ = self.act(s, mode="evaluate")
            s, reward, terminated, truncated, info = self.env.step(a)
            rewards += reward
            if terminated or truncated:
                break
        return rewards
    
    def learning_curve(self):
        result_path = self._check_dir()
        x = range(len(self.epoch_rewards))
        y = self.epoch_rewards
        plt.plot(x, y)
        name = "REINFORCE_" + self.env.spec.id
        plt.title(name)
        plt.savefig(f'{os.path.join(result_path, name+".png")}', bbox_inches='tight')

    def save(self):
        result_path = self._check_dir()
        torch.save(self.state_dict(), os.path.join(result_path, f"REINFORCE_{self.env.spec.id}_h{self.h_size}.pth"))

    def _check_dir(self):
        file_path = __file__
        file_name = os.path.basename(file_path).split(".")[0]
        result_path = os.path.join("results", file_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        return result_path

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    agent = REINFORCE(env, epochs=1000).to(device)
    agent.train()
    agent.learning_curve()
    agent.save()