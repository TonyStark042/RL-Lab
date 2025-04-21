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
from core.args import REINFORCEArgs


class REINFORCE(PRL):
    def __init__(self, env, args:REINFORCEArgs):       
        super().__init__(env=env, args=args, model_names=["policy_net"])
        action_shape = self.action_dim if self.has_continuous_action_space else self.action_num
        self.policy_net = Policy_net(self.state_dim, action_shape, self.h_size, self.has_continuous_action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def act(self, state, mode:Literal["train", "evaluate", "test"] = "train"):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        dist = self.policy_net(state)
        action = dist.sample()   
        if mode == "train":  # evaluate ensures only return the action                            
            return action.detach().cpu().numpy() , dist.log_prob(action)
        elif mode == "evaluate":
            return action.detach().cpu().numpy()
        else:
            action = dist.probs.argmax(dim=-1)
            return action.detach().cpu().numpy()

    def train(self):
        start = time.time()
        reach_maxTimestep = False
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            self.log_probs = []
            self.rewards = []
            s = self.env.reset()[0]

            while True:
                a, log_prob = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a.squeeze())
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
            self.epoch += 1

            if self.train_mode == "episode":
                early_stop = self.monitor.epoch_report()
                
            if early_stop or reach_maxTimestep:
                break
        end = time.time()
        self.training_time += (end - start)

    def _update(self):
        policy_reward = torch.tensor(0.0).to(self.device)

        returns = []
        next_return = 0.0  
        for reward in self.rewards[::-1]:
            cur_return = reward + self.gamma * next_return
            next_return = cur_return
            returns.append(cur_return)
        returns.reverse()
        for log_prob, disc_return in zip(self.log_probs, returns):
            policy_reward += (-log_prob.sum(-1) * (disc_return - self.baseline)).sum()       # log_prob * disc_return，but the default gradient descent direction is "-"，Adding "-" means gradient rise.
                                       
        self.optimizer.zero_grad()
        policy_reward.backward()
        self.optimizer.step()