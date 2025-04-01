import os
import time
from typing import Literal
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical       
from collections import deque
from core.module import PRL
from core.net import Policy_net
from core.args import PRLArgs


class REINFORCE(PRL):
    def __init__(self, env, args):       
        super().__init__(env=env, args=args, model_name="policy_net")
        self.policy_net = Policy_net(self.state_num, self.action_num, self.h_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def act(self, state, mode:Literal["train", "evaluate"] = "train"):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state).squeeze()
        prob_dist = Categorical(probs)
        action = prob_dist.sample()   
        if mode == "train":  # evaluate ensures only return the action                            
            return action.item(), prob_dist.log_prob(action)
        elif mode == "evaluate":
            return action.item()

    def train(self):
        start = time.time()
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            self.log_probs = []
            self.rewards = []
            s = self.env.reset()[0]

            while True:
                a, log_prob = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                self.timestep += 1
                self.log_probs.append(log_prob) 
                self.rewards.append(reward)
                
                if self.train_mode == "timestep":
                    early_stop = self.monitor.timestep_report()
                    reach_maxTimestep = self.timestep >= self.max_timesteps
                    if early_stop or reach_maxTimestep:
                        break

                if terminated or truncated:
                    self.epoch_record.append(sum(self.rewards))
                    break

            self._update()

            if self.train_mode == "episode":
                early_stop = self.monitor.epoch_report()
                self.epoch += 1

            if early_stop or reach_maxTimestep:
                break
        end = time.time()
        self.training_time += (end - start)

    def _update(self):
        policy_reward = torch.tensor(0.0).to(self.device)
        steps = len(self.rewards)
        returns = deque()   

        for t in range(steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)   
            returns.appendleft(self.gamma*disc_return_t + self.rewards[t])  # Dynamic Programming
        for log_prob, disc_return in zip(self.log_probs, returns):
            policy_reward += (-log_prob * (disc_return - self.baseline)).sum()       # log_prob * disc_return，but the default gradient descent direction is "-"，Adding "-" means gradient rise.
                                       
        self.optimizer.zero_grad()
        policy_reward.backward()
        self.optimizer.step()