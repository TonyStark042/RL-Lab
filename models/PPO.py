import os
from typing import Literal
import gymnasium as gym
import torch
from torch.distributions import Categorical       
from core.module import PRL, PRLArgs
from core.buffer import ReplayBuffer
from core.args import PRLArgs
from core.net import ActorCritic
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np

class PPO(PRL):
    def __init__(self, env, args: PRLArgs):
        super().__init__(env=env, args=args, alg_name="PPO", model_name="trg_policy",)
        self.buffer = ReplayBuffer()
        self.trg_policy = ActorCritic(self.state_num, self.action_num, self.h_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.trg_policy.actor.parameters(), self.actor_lr)
        self.critic_optimizer =  torch.optim.Adam(self.trg_policy.critic.parameters(), self.critic_lr)

        self.act_policy = ActorCritic(self.state_num, self.action_num, self.h_size).to(self.device)
        self.act_policy.load_state_dict(self.trg_policy.state_dict())
                                                                    
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def act(self, state, mode: Literal["train", "evaluate"] = "train"): 
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        probs = self.act_policy.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def train(self):
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            cur_s = self.env.reset()[0]
            self.rewards = []
            while True:
                a = self.act(cur_s)
                next_s, reward, terminated, truncated, info = self.env.step(a)
                if self.is_gae:
                    trainsition = (cur_s, next_s, a, reward, terminated or truncated)
                else:
                    trainsition = (cur_s, None, a, reward, terminated or truncated)
                self.buffer.add(trainsition)

                cur_s = next_s
                self.timestep += 1
                self.rewards.append(reward)
                
                if self.timestep % self.update_freq == 0 and len(self.buffer) > 0:
                    self._update()
                
                if self.train_mode == "timestep":
                    early_stop = self.monitor.timestep_report()
                    if early_stop:
                        break  # exit inner loop first

                if terminated or truncated:
                    self.epoch_record.append(sum(self.rewards))
                    break
            
            if self.train_mode == "episode":
                early_stop = self.monitor.epoch_report()
                self.epoch += 1

            if early_stop:
                break
    
    def _update(self): 
        old_states, old_next_states, old_actions, rewards, is_terminals = self.buffer.sample_all()
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float) # Creating a tensor a list of numpy.ndarrays is extremely slow, eg: [np.array, np.array, np.array]
        old_actions = torch.tensor(old_actions, device=self.device).unsqueeze(-1)

        old_log_probs = torch.log(self.act_policy.actor(old_states).gather(1, old_actions)).detach()
        old_state_values = self.act_policy.critic(old_states)
        
        if self.is_gae:
            is_terminals = torch.tensor(is_terminals, device=self.device).int().unsqueeze(-1)
            old_next_states = torch.tensor(np.array(old_next_states), device=self.device, dtype=torch.float)
            old_next_states_values = self.act_policy.critic(old_next_states)
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).unsqueeze(-1)
            td_target = rewards + self.gamma * old_next_states_values * (1 - is_terminals)
            td_delta = td_target - old_state_values
            advantages = self.gae(td_delta.cpu()).to(self.device).detach()   # 使用GAE估计的Advantage
        else:
            returns = []
            for reward, is_terminal in zip(rewards[::-1], is_terminals[::-1]):
                if is_terminal:
                    next_return = 0
                else:
                    next_return = 0 if len(returns) == 0 else returns[-1]
                cur_return = reward + (self.gamma * next_return)
                returns.append(cur_return)
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float, device=self.device).unsqueeze(-1)  # 蒙特卡洛估计的Advantage，高方差
            # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            advantages = returns - old_state_values.detach()

        for _ in range(self.update_times):            
            dist = Categorical(self.trg_policy.actor(old_states))
            log_probs = dist.log_prob(old_actions.squeeze()).unsqueeze(-1)
            dist_entropy = dist.entropy().mean()
            state_values = self.trg_policy.critic(old_states)

            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist_entropy.mean()
            if self.is_gae:
                critic_loss = self.criterion(state_values, td_target.detach()).mean()
            else:
                critic_loss = self.criterion(state_values, returns).mean()
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.trg_policy.parameters(), 5) # imporrtant, avoid gradient explosion
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        self.act_policy.load_state_dict(self.trg_policy.state_dict())
        # print(f"Target_policy has updated, actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}")
        self.buffer.clear()

        return actor_loss.item(), critic_loss.item()
    
    def test(self):
        result_dir = self.monitor._check_dir()
        para = os.path.join(result_dir, f"weight.pth")
        self.logger.info(f"Loading model from {para}")
        self.act_policy.load_state_dict(torch.load(para))
        rewards = self.evaluate()
        self.logger.info(f"{self.alg_name} test reward in {self.env_name}: {rewards}")
    