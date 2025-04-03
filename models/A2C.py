import time
from typing import Literal
from core.net import ActorCritic
from core.module import PRL
from core.args import PRLArgs
from torch import optim
import torch
import gymnasium as gym
import os
import numpy as np

class A2C(PRL):
    def __init__(self, env, args:PRLArgs) -> None:
        super().__init__(env, args=args, model_name="model")
        self.model = ActorCritic(self.state_num, self.action_num, self.h_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.critic_lr)

    def act(self, state, mode:Literal["train", "evaluate"]="train"):
        state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
        dist, value = self.model(state)
        action = dist.sample()
        action = self.adapt_action(action)
        if mode == "train":
            return action.cpu().numpy(), value, dist
        else:
            return action.cpu().numpy()
        
    def train(self):
        start = time.time()
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            self.log_probs = []
            self.rewards = []
            self.epoch_values = []
            self.epoch_entropy = 0
            s = self.env.reset()[0]

            while True:
                a, v, dist = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                self.timestep += 1
                self.log_probs.append(dist.log_prob(torch.tensor(a, device=self.device)))
                self.rewards.append(reward)
                self.epoch_values.append(v)
                self.epoch_entropy += dist.entropy().mean()

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
        returns = []
        for reward in self.rewards[::-1]:
            returns.insert(0, self.gamma * (0 if len(returns) ==0 else returns[0]) + reward)
        returns = torch.tensor(returns, device=self.device).detach()
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.epoch_values)
        advantage = returns - values  # To train the critic, be closer to the collcted return, so that better estimate the action advantage. 
        critic_loss = advantage.pow(2).mean()
        actor_loss  = -(log_probs * advantage.detach()).mean()
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * self.epoch_entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item()