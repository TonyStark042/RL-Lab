from core.net import ActorCritic
from core.rollout import OnPolicy
from core.baseModule import PRL
from core.args import A2CArgs
from torch import optim
import torch
import numpy as np
from core.buffer import HorizonBuffer

class A2C(OnPolicy[A2CArgs]):
    def __init__(self, env, args:A2CArgs) -> None:
        super().__init__(env, args=args, model_names=["model"])
        self.model = ActorCritic(self.env, self.cfg.h_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.cfg.critic_lr)
        self.buffer = HorizonBuffer(horizon=self.cfg.horizon)

    def act(self, state, deterministic=False):
        state = torch.tensor(state, device=self.device, dtype=torch.float).reshape(-1, self.env.state_dim)
        dist = self.model.actor(state)
        if deterministic:
            if self.env.has_continuous_action_space:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.detach().cpu().numpy()

    def _update(self):
        states, actions, rewards, _, dones = self._sample_all(clear=True)
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device)
        dists = self.model.actor(states)
        log_probs = dists.log_prob(actions).reshape(-1, self.env.action_dim).sum(-1, keepdim=True)
        entropys = dists.entropy().reshape(-1, 1)
        values = self.model.critic(states).reshape(-1, 1)

        returns = []
        next_return = 0.0
        for reward, done  in zip(rewards[::-1], dones[::-1]):
            cur_return = reward + self.cfg.gamma * next_return * (1 - done)
            next_return = cur_return
            returns.append(cur_return)
        returns.reverse()
        returns = torch.tensor(returns, device=self.device).detach().reshape(-1, 1)
        advantage = returns - values
        critic_loss = advantage.pow(2).mean()
        actor_loss  = -(log_probs * advantage.detach()).mean()
        loss = actor_loss + 0.5 * critic_loss - self.cfg.entropy_coef * entropys.mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.buffer.clear()

        return {"loss":loss.item(), "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}  