import torch
from core.baseModule import RL
from core.buffer import ReplayBuffer
from core.net import Critic_Qnet, Determin_PolicyNet
from torch import nn
import numpy as np
from core.rollout import OffPolicy
import torch.nn.functional as F


class TD3(OffPolicy):
    def __init__(self, env, args):
        super().__init__(env=env, args=args, model_names=["actor", "critic1"],)
        self.buffer = ReplayBuffer(capacity=self.cfg.memory_size)
        self.noise = OUNoise(env.action_space, decay_period=self.cfg.max_timesteps) if self.cfg.noise_type == "OU" else GaussianNoise(env.action_space)
        self.actor = Determin_PolicyNet(self.env, self.cfg.h_size).to(self.device)
        self.critic1 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.critic2 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.trg_actor = Determin_PolicyNet(self.env, self.cfg.h_size).to(self.device)
        self.trg_critic1 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.trg_critic2 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.trg_actor.load_state_dict(self.actor.state_dict())
        self.trg_critic1.load_state_dict(self.critic1.state_dict())
        self.trg_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.cfg.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), self.cfg.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), self.cfg.critic_lr)
    
    @torch.no_grad()
    def act(self, state, deterministic=False): 
        state = torch.FloatTensor(state).to(self.device).reshape(-1, self.env.state_dim)
        action = self.actor(state).cpu().numpy()
        action = self.a2a(action)
        if not deterministic:
            action = self.noise.get_action(action)
        return action

    def _update(self):
        if len(self.buffer) < self.cfg.expl_steps:
            return {}
        state, action, reward, next_state, done = self._sample(self.cfg.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).reshape(-1, 1)
        done = torch.FloatTensor(done).to(self.device).reshape(-1, 1)

        # update critic
        Q1_value, Q2_value = self.critic1(state, action), self.critic2(state, action)
        next_action = self.trg_actor(next_state)
        next_Q1_value, next_Q2_value = self.trg_critic1(next_state, next_action), self.trg_critic2(next_state, next_action)
        next_Q = torch.min(next_Q1_value, next_Q2_value)
        target_Q_value = reward + (1 - done) * self.cfg.gamma * next_Q
        critic1_loss = F.mse_loss(Q1_value, target_Q_value.detach())
        critic2_loss = F.mse_loss(Q2_value, target_Q_value.detach())
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # update actor
        if self.timestep.sum() % 2 == 0:
            curr_action = self.actor(state)  # Current actor's action, can't use buffer's action  # 这里输入了一个大的state，导致输出趋向边界值,导致critic一直对边界值评价
            actor_loss = -self.critic1(state, curr_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        else:
            actor_loss = np.array([0.0])

        # soft update
        for param, target_param in zip(self.critic1.parameters(), self.trg_critic1.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.trg_critic2.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.trg_actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        return {"actor_Q": actor_loss.item(), "critic1_loss": critic1_loss.item()}

class OUNoise(object):
    '''
    Ornstein–Uhlenbeck noise
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu 
        self.theta = theta 
        self.sigma = max_sigma 
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.n_actions = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu

    def generate_noise(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs
    
    def get_action(self, action, t=0):
        ou_obs = self.generate_noise()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)

class GaussianNoise:
    def __init__(self, action_space, mu=0.0, std=0.1, decay_factor=0.999, min_std=0.0001, clip=0.3):
        self.mu = mu
        self.std = std
        self.initial_std = std
        self.decay_factor = decay_factor
        self.min_std = min_std
        self.low = action_space.low
        self.high = action_space.high
        self.shape = action_space.shape
        self.clip = clip

    def reset(self):
        self.std = self.initial_std

    def generate_noise(self):
        noise = np.random.normal(self.mu, self.std, size=self.shape)
        noise = np.clip(noise, -self.clip, self.clip)
        return noise
    
    def get_action(self, action, t=0):
        noise = self.generate_noise()
        # Decay std over time
        self.std = max(self.min_std, self.std * self.decay_factor)
        return np.clip(action + noise, self.low, self.high)