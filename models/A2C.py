import time
from typing import Literal
from core.net import ActorCritic
from core.module import PRL
from core.args import PRLArgs
from torch import optim
import torch
import gymnasium as gym
import numpy as np
from core.buffer import ReplayBuffer

class A2C(PRL):
    def __init__(self, env, args:PRLArgs) -> None:
        super().__init__(env, args=args, model_name="model")
        self.model = ActorCritic(self.state_num, self.action_num, self.h_size, self.has_continuous_action_space).to(self.device)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.critic_lr)
        self.buffer = ReplayBuffer()

    def act(self, state, mode:Literal["train", "evaluate", "test"]="train"):
        state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
        dist = self.model.actor(state)
        action = dist.sample()
        if mode == "train":
            value = self.model.critic(state)
            action = dist.sample()
            return action.detach().cpu().numpy(), dist, dist.log_prob(action), value
        elif mode == "evaluate":
            return action.detach().cpu().numpy()
        else:
            if self.has_continuous_action_space:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
            return action.detach().cpu().numpy()
        
    def train(self):
        start = time.time()
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            s = self.env.reset()[0]
            rewards = []

            while True:
                a, dist, log_prob, value = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                self.timestep += 1
                rewards.append(reward)
                transition = (value, log_prob, dist.entropy().mean(), reward, 1 if terminated or truncated else 0)
                self.buffer.add(transition)

                if self.timestep % self.horizon == 0 and len(self.buffer) > 0:
                    self._update()

                if self.train_mode == "timestep":
                    early_stop = self.monitor.timestep_report()
                    reach_maxTimestep = self.timestep >= self.max_timesteps
                    if early_stop or reach_maxTimestep:
                        break
                    
                if terminated or truncated:
                    self.epoch_record.append(sum(rewards))
                    break

            self.epoch += 1

            if self.train_mode == "episode":
                early_stop = self.monitor.epoch_report()
                reach_maxTimestep = False
                
            if early_stop or reach_maxTimestep:
                break
        end = time.time()
        self.training_time += (end - start)

    def _update(self):
        values, log_probs, entropys, rewards, dones = self.buffer.sample_all()
        returns = []
        next_return = 0.0
        for reward, done  in zip(rewards[::-1], dones[::-1]):
            cur_return = reward + self.gamma * next_return * done
            next_return = cur_return
            returns.append(cur_return)
        returns.reverse()
        returns = torch.tensor(returns, device=self.device).detach().unsqueeze(-1)
        log_probs = torch.stack(log_probs, dim=0)
        log_probs = log_probs.sum(-1, keepdim=True)
        values = torch.stack(values, dim=0)
        advantage = returns - values
        critic_loss = advantage.pow(2).mean()
        actor_loss  = -(log_probs * advantage.detach()).mean()
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * sum(entropys)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.buffer.clear()

        return loss.item(), actor_loss.item(), critic_loss.item()