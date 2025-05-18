import torch
from core.buffer import ReplayBuffer
from core.net import Critic_Qnet, Policy_net, SAC_PolicyNet
import numpy as np
from core.rollout import OffPolicy
import torch.nn.functional as F
from core.args import SACArgs


class SAC(OffPolicy[SACArgs]):
    def __init__(self, env, args):
        super().__init__(env=env, args=args, model_names=["actor", "critic1", "critic2"],)
        self.buffer = ReplayBuffer(capacity=self.cfg.memory_size)
        self.actor = SAC_PolicyNet(self.env, self.cfg.h_size).to(self.device)
        self.critic1 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.critic2 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.trg_critic1 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.trg_critic2 = Critic_Qnet(self.env, self.cfg.h_size).to(self.device)
        self.trg_critic1.load_state_dict(self.critic1.state_dict())
        self.trg_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.cfg.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), self.cfg.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), self.cfg.critic_lr)

        self.target_entropy = torch.tensor(-self.env.action_dim, dtype=float, requires_grad=True, device=self.device)
        self.alpha = self.cfg.alpha
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.critic_lr)
        
    def act(self, state, deterministic=False, log_probs=False): 
        state = torch.FloatTensor(state).to(self.device).reshape(-1, self.env.state_dim)
        dist:torch.distributions.Normal = self.actor(state)
        if deterministic:
            u = dist.mean
        else:
            u = dist.rsample()
        action = torch.tanh(u)
        action = self.a2a(action)
        if log_probs:
            # logp_pi_a = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
            log_prob = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
            # log_prob = log_prob.clamp(min=-10, max=0)
            return action, log_prob
        else:
            return action.detach().cpu().numpy()

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
        with torch.no_grad():
            next_action, next_log_probs = self.act(next_state.cpu().numpy(), deterministic=False, log_probs=True)
            next_Q1_value, next_Q2_value = self.trg_critic1(next_state, next_action), self.trg_critic2(next_state, next_action)
            next_Q = torch.min(next_Q1_value, next_Q2_value)
            target_Q_value = reward + (1 - done) * self.cfg.gamma * (next_Q - self.alpha * next_log_probs)

        Q1_value, Q2_value = self.critic1(state, action), self.critic2(state, action)
        critic_loss = F.mse_loss(Q1_value, target_Q_value.detach()) + F.mse_loss(Q2_value, target_Q_value.detach())
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # update actor
        if self.timestep.sum() % 2 == 0:
            for params in self.critic1.parameters(): params.requires_grad = False
            for params in self.critic2.parameters(): params.requires_grad = False

            cur_action, log_probs = self.act(state.cpu().numpy(), deterministic=False, log_probs=True)  # Policy based, should use current actor's action, can't use buffer's action
            Q1_value, Q2_value = self.critic1(state, cur_action), self.critic2(state, cur_action)
            Q =  torch.min(Q1_value, Q2_value)
            a_loss = (self.alpha * log_probs - Q).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            for params in self.critic1.parameters(): params.requires_grad = True
            for params in self.critic2.parameters(): params.requires_grad = True

            # update alpha
            with torch.no_grad():
                _, log_probs = self.act(state.cpu().numpy(), deterministic=False, log_probs=True)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            a_loss = np.array([0.0])

        # soft update
        for param, target_param in zip(self.critic1.parameters(), self.trg_critic1.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.trg_critic2.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        return {"actor_loss": a_loss.item(), "critic_loss": critic_loss.item()}