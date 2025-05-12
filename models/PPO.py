import torch
from core.buffer import HorizonBuffer
from core.args import PRLArgs
from core.net import ActorCritic
from torch import nn
import numpy as np
from core.rollout import OnPolicy
from core.args import PPOArgs

class PPO(OnPolicy[PPOArgs]):
    def __init__(self, env, args):
        super().__init__(env=env, args=args, model_names=["act_policy"],)
        self.buffer = HorizonBuffer(horizon=self.cfg.horizon)
        self.trg_policy = ActorCritic(self.env, self.cfg.h_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.trg_policy.actor.parameters(), self.cfg.actor_lr)
        self.critic_optimizer =  torch.optim.Adam(self.trg_policy.critic.parameters(), self.cfg.critic_lr)

        self.act_policy = ActorCritic(self.env, self.cfg.h_size).to(self.device)
        self.act_policy.load_state_dict(self.trg_policy.state_dict())
                                                                    
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def act(self, state, deterministic=False): 
        state = torch.FloatTensor(state).to(self.device).reshape(-1, self.env.state_dim)
        dist = self.act_policy.actor(state)
        if deterministic:
            if self.env.has_continuous_action_space:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.cpu().numpy()
    
    def _update(self, shuffle=True):
        self.cfg.entropy_coef *= self.cfg.entropy_decay

        old_states, old_actions, rewards, old_next_states, is_terminals = self._sample_all(clear=True)
        old_states = torch.tensor(old_states, device=self.device, dtype=torch.float)
        old_actions = torch.tensor(old_actions, device=self.device)
        old_log_probs = self.act_policy.actor(old_states).log_prob(old_actions).detach().reshape(-1, self.env.action_dim) # getting log_probs from Categorical distribution need 1 dim
        old_state_values = self.act_policy.critic(old_states).detach()

        is_terminals = torch.tensor(is_terminals, device=self.device).int().unsqueeze(-1)
        old_next_states = torch.tensor(old_next_states, device=self.device, dtype=torch.float)
        old_next_states_values = self.act_policy.critic(old_next_states).detach()
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).reshape(-1, 1)
        td_target = rewards + self.cfg.gamma * old_next_states_values * (1 - is_terminals)
        td_delta = td_target - old_state_values
        advantages = self.gae(td_delta.cpu(), is_terminals.cpu()).to(self.device).detach()
        if self.cfg.norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        gae_returns = advantages + old_state_values
        
        assert (old_states.ndim == 2 and old_log_probs.ndim == 2) and (old_state_values.shape == advantages.shape), f"old_states shape: {old_states.shape}, old_actions shape: {old_actions.shape}, old_log_probs shape: {old_log_probs.shape}, old_state_values shape: {old_state_values.shape}, advantages shape: {advantages.shape}"
        total_samples = old_states.shape[0]

        approx_kl_divs = []
        continue_training = True
        
        for update_time in range(self.cfg.update_times):
            if not continue_training:
                break
            if shuffle:
                indices = torch.randperm(total_samples)
            else:
                indices = torch.arange(total_samples)

            for start in range(0, total_samples, self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, total_samples)
                batch_indices = indices[start:end]
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_state_values = old_state_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = gae_returns[batch_indices]

                dist = self.trg_policy.actor(batch_states)
                log_probs = dist.log_prob(batch_actions).reshape(-1, self.env.action_dim)
                dist_entropy = dist.entropy().mean()
    
                log_ratio = log_probs.sum(-1, keepdim=True) - batch_old_log_probs.sum(-1, keepdim=True)
                ratios = torch.exp(log_ratio)
                
                with torch.no_grad():
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.cfg.target_kl is not None and approx_kl_div > self.cfg.target_kl:
                    continue_training = False
                    self.logger.info(f"Early stopping updating at update_time {update_time+1} due to KL divergence exceeding KL threshold {self.cfg.target_kl}.")
                    break
                
                assert ratios.shape == batch_advantages.shape, f"ratios shape: {ratios.shape}, advantages shape: {batch_advantages.shape}"
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * dist_entropy
                
                state_values = self.trg_policy.critic(batch_states)
                value_clipped = batch_old_state_values + torch.clamp(state_values - batch_old_state_values, -self.cfg.eps_clip, self.cfg.eps_clip)
                
                critic_loss1 = self.criterion(state_values, batch_returns)
                critic_loss2 = self.criterion(value_clipped, batch_returns)
                critic_loss = torch.max(critic_loss1, critic_loss2).mean() * 0.5

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
        kl_mean = np.mean(approx_kl_divs) if approx_kl_divs else 0
        return {"actor_loss":actor_loss.item(), "critic_loss":critic_loss.item(), "KL":kl_mean.item()}
    