import os
import time
from typing import Literal
import gymnasium as gym
import torch
from torch.distributions import Categorical,MultivariateNormal      
from core.module import PRL, PRLArgs
from core.buffer import ReplayBuffer
from core.args import PRLArgs
from core.net import ActorCritic
from torch import nn
import numpy as np
from utils import normalize

class PPO(PRL):
    def __init__(self, env, args: PRLArgs):
        super().__init__(env=env, args=args, model_names=["act_policy"],)
        self.buffer = ReplayBuffer()
        action_shape = self.action_dim if self.has_continuous_action_space else self.action_num
        self.trg_policy = ActorCritic(self.state_dim, action_shape, self.h_size, self.has_continuous_action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.trg_policy.actor.parameters(), self.actor_lr)
        self.critic_optimizer =  torch.optim.Adam(self.trg_policy.critic.parameters(), self.critic_lr)

        self.act_policy = ActorCritic(self.state_dim, action_shape, self.h_size, self.has_continuous_action_space).to(self.device)
        self.act_policy.load_state_dict(self.trg_policy.state_dict())
                                                                    
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def act(self, state, mode: Literal["train", "evaluate", "test"] = "train"): 
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.act_policy.actor(state)
        if mode == "train" or mode == "evaluate":
            action = dist.sample()
        else:
            if self.has_continuous_action_space:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        return action.cpu().numpy()

    def train(self):
        start = time.time()
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            cur_s = self.env.reset()[0]
            rewards = []
            while True:
                a = self.act(cur_s)
                next_s, reward, terminated, truncated, info = self.env.step(a.squeeze())
                if self.is_gae:
                    trainsition = (cur_s, next_s, a, reward, terminated or truncated)
                else:
                    trainsition = (cur_s, None, a, reward, terminated or truncated)
                self.buffer.add(trainsition)

                cur_s = next_s
                self.timestep += 1
                rewards.append(reward)
                
                if self.timestep % self.horizon == 0 and len(self.buffer) > 0:
                    self._update()
                
                if self.train_mode == "timestep":
                    early_stop = self.monitor.timestep_report()
                    reach_maxTimestep = self.timestep >= self.max_timesteps
                    if early_stop or reach_maxTimestep:
                        break  # exit inner loop first

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
    
    def _update(self, shuffle=True):
        self.entropy_coef *= self.entropy_decay

        old_states, old_next_states, old_actions, rewards, is_terminals = self.buffer.sample_all()
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float)
        # old_states = normalize(old_states)
        old_actions = torch.tensor(np.array(old_actions), device=self.device)
        old_log_probs = self.act_policy.actor(old_states).log_prob(old_actions.squeeze()).detach()
        old_log_probs = old_log_probs.unsqueeze(-1) if old_log_probs.ndim == 1 else old_log_probs
        old_state_values = self.act_policy.critic(old_states).detach()
        
        if self.is_gae:
            is_terminals = torch.tensor(is_terminals, device=self.device).int().unsqueeze(-1)
            old_next_states = torch.tensor(np.array(old_next_states), device=self.device, dtype=torch.float)
            old_next_states_values = self.act_policy.critic(old_next_states).detach()
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).unsqueeze(-1)
            td_target = rewards + self.gamma * old_next_states_values * (1 - is_terminals)
            td_delta = td_target - old_state_values
            advantages = self.gae(td_delta.cpu(), is_terminals.cpu()).to(self.device).detach() 
        else:
            returns = []
            next_return = 0.0
            for reward, is_terminal in zip(rewards[::-1], is_terminals[::-1]):
                cur_return = reward + (self.gamma * next_return * is_terminal)
                next_return = cur_return
                returns.append(cur_return)
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float, device=self.device).unsqueeze(-1)  # Monte-Carlo estimation, high variance
            advantages = returns - old_state_values
        # advantages = (advantages - advantages.mean()) / ((advantages.std()+1e-4))
        
        assert (old_states.ndim == 2 and old_actions.ndim == 2) and (old_log_probs.shape == old_actions.shape) and (old_state_values.shape == advantages.shape), f"old_states shape: {old_states.shape}, old_actions shape: {old_actions.shape}, old_log_probs shape: {old_log_probs.shape}, old_state_values shape: {old_state_values.shape}, advantages shape: {advantages.shape}"
        total_samples = old_states.shape[0]

        for _ in range(self.update_times):
            if shuffle:         
                indices = torch.randperm(total_samples) # randomly shuffle the indices
            else:
                indices = np.arange(total_samples)
            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_indices = indices[start:end]
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_state_values = old_state_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                dist = self.trg_policy.actor(batch_states)
                log_probs = dist.log_prob(batch_actions.squeeze())
                log_probs = log_probs.unsqueeze(-1) if log_probs.ndim == 1 else log_probs
                dist_entropy = dist.entropy().mean()
                # ratios = torch.exp(log_probs.sum(-1, keepdim=True) - torch.clamp(batch_old_log_probs.sum(-1, keepdim=True), -10, 2))
                ratios = torch.exp(log_probs.sum(-1, keepdim=True) - batch_old_log_probs.sum(-1, keepdim=True))
                assert ratios.shape == batch_advantages.shape, f"ratios shape: {ratios.shape}, advantages shape: {batch_advantages.shape}"
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist_entropy
                
                state_values = self.trg_policy.critic(batch_states)
                value_clipped = batch_old_state_values + torch.clamp(state_values - batch_old_state_values, -self.eps_clip, self.eps_clip)
                if self.is_gae:
                    batch_td_target = td_target[batch_indices]
                    critic_loss1 = self.criterion(state_values, batch_td_target)
                    critic_loss2 = self.criterion(value_clipped, batch_td_target)
                else:
                    batch_returns = returns[batch_indices]
                    critic_loss1 = self.criterion(state_values, batch_returns)
                    critic_loss2 = self.criterion(value_clipped, batch_returns)
                critic_loss = torch.max(critic_loss1, critic_loss2).mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
            
                torch.nn.utils.clip_grad_norm_(self.trg_policy.actor.parameters(), max_norm=5, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.trg_policy.critic.parameters(), max_norm=5, norm_type=2)
                # self.monitor.grad_report(self.trg_policy.actor)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
            
        self.act_policy.load_state_dict(self.trg_policy.state_dict())
        self.buffer.clear()

        return actor_loss.item(), critic_loss.item()
    