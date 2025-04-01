import time
from typing import Literal
from core.module import VRL
from core.net import Q_net, Dueling_Q_net
from torch import optim
import torch
import numpy as np
from torch import nn
import gymnasium as gym
from argparse import ArgumentParser
from core.args import VRLArgs
from core.buffer import ReplayBuffer
import os


class DQN(VRL):
    def __init__(self, env, args:VRLArgs=None):
        super().__init__(env, args=args, model_name="policy_net",)
        if "Dueling" in self.alg_name:
            self.policy_net = Dueling_Q_net(self.state_num, self.action_num, self.h_size, self.noise, self.std_init).to(self.device)
            self.target_net = Dueling_Q_net(self.state_num, self.action_num, self.h_size, self.noise, self.std_init).to(self.device)
        else:
            self.policy_net = Q_net(self.state_num, self.action_num, self.h_size, self.noise).to(self.device)
            self.target_net = Q_net(self.state_num, self.action_num, self.h_size, self.noise).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayBuffer(self.memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) # only policy_net need optimizer
    
    @torch.no_grad() # Based on Q value to select action, no need to calculate gradient
    def act(self, state, mode:Literal["train", "evaluate"]="train"):
        if mode == "train":
            if self.noise:
                state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
                action = self.policy_net(state).argmax().item()
            else:
                action = self.epsilon_greedy(state)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.policy_net(state).argmax().item()
        return action

    def train(self):
        start = time.time()
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            rewards = []
            s = self.env.reset()[0]
            while True:
                a = self.act(s)
                next_s, reward, terminated, truncated, info = self.env.step(a)
                self.timestep += 1
                rewards.append(reward)
                self.memory.add((s, a, reward, next_s, terminated)) 
                s = next_s
                loss = self._update()
                if self.timestep % self.sync_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if self.train_mode == "timestep":
                    if self.noise:
                        early_stop = self.monitor.timestep_report(loss=loss) 
                    else:
                        early_stop = self.monitor.timestep_report(loss=loss, epsilon=self.epsilon)
                    reach_maxTimestep = self.timestep >= self.max_timesteps
                    if early_stop or reach_maxTimestep:
                        break 
        
                if terminated or truncated:
                    self.epoch_record.append(sum(rewards))
                    break

            if self.noise:
                self.policy_net.reset_noise()
                self.target_net.reset_noise()
                if self.train_mode == "episode":
                    early_stop = self.monitor.epoch_report(loss=loss, weight_epsilon=self.policy_net.fc2.weight_epsilon.mean().item(), bias_epioslon=self.policy_net.fc2.bias_epsilon.mean().item())
                    self.epoch += 1
            else:
                if self.train_mode == "episode":
                    early_stop = self.monitor.epoch_report(loss=loss, epsilon=self.epsilon)
                    self.epoch += 1
            
            if early_stop or reach_maxTimestep:
                break
        end = time.time()
        self.training_time += (end - start)

    def _update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)  
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)  
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.float32(dones), device=self.device)

        Q_values = self.policy_net(states).gather(dim=1, index=actions)
        
        if "Double" in self.alg_name:
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True) # Q_net select next action, but evaluated by target_net
            next_Q = self.target_net(next_states).gather(1, next_actions).squeeze()
        else:
            next_Q = self.target_net(next_states).max(dim=1)[0].detach()            
        
        targets = rewards + self.gamma * next_Q * (1 - dones)
        loss = nn.MSELoss()(Q_values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()