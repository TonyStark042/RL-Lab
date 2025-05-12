from core.buffer import EpisodeBuffer
from typing import Literal
import gymnasium as gym
import torch
import torch.optim as optim       
from core.rollout import OnPolicy
from core.net import Policy_net
from core.args import REINFORCEArgs
import numpy as np

class REINFORCE(OnPolicy):
    def __init__(self, env, args:REINFORCEArgs):       
        super().__init__(env=env, args=args, model_names=["policy_net"])
        action_shape = self.env.action_dim if self.env.has_continuous_action_space else self.env.action_num
        self.policy_net = Policy_net(self.env, self.cfg.h_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.buffer = EpisodeBuffer(capacity=self.cfg.max_episode_steps)
    
    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).float().reshape(-1, self.env.state_dim).to(self.device)
        dist = self.policy_net(state)
        if deterministic:
            if self.env.has_continuous_action_space:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.detach().cpu().numpy()

    def _update(self):
        policy_reward = torch.tensor(0.0).to(self.device)

        states, actions, rewards, _, _ = self._sample_all(clear=True)
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(actions), device=self.device)
        log_probs = self.policy_net(states).log_prob(actions)

        returns = []
        next_return = 0.0  
        for reward in rewards[::-1]:
            cur_return = reward + self.cfg.gamma * next_return
            next_return = cur_return
            returns.append(cur_return)
        returns.reverse()
        for log_prob, disc_return in zip(log_probs, returns):
            policy_reward += (-log_prob.sum(-1) * (disc_return - self.cfg.baseline)).sum()       # log_prob * disc_return，but the default gradient descent direction is "-"，Adding "-" means gradient rise.
                                       
        self.optimizer.zero_grad()
        policy_reward.backward()
        self.optimizer.step()

        return {"Expecred reward": policy_reward.item()}