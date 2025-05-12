import numpy as np
from core.args import *
import time
from core.baseModule import PRL, VRL
from typing import Generic, TypeVar

P1 = TypeVar('Onpolicy', bound='PRLArgs')
class OnPolicy(PRL[P1]):
    def __init__(self, env, args:P1, **kwargs):
        super().__init__(env, args, **kwargs)
    
    def train(self):
        start = time.time()
        report_items = {}
        cur_s, _ = self.env.reset()
        while self.episode.sum() < self.cfg.max_epochs and self.timestep.sum() < self.cfg.max_timesteps:
            next_s, action, reward, done, info = self.step(self.env, cur_s)
            self.buffer.add((cur_s, action, reward, next_s, done))
            self.timestep += 1
            
            if done.any():
                next_s = self.reset(done)
            cur_s = next_s

            if self.buffer.is_ready(done=done, timestep=self.timestep.sum()): 
                report_items = self._update()

            early_stop = self.monitor.timestep_report(report_items)
            if early_stop:
                    break
            
            end = time.time()
            self.training_time = (end - start)

    def _sample_all(self, clear=True):
        states, actions, rewards, next_states, is_terminals = self.buffer.sample_all(clear=clear)
        is_terminals = list(is_terminals)
        is_terminals[-1] = np.array([True] * self.cfg.num_envs)
        states, actions, rewards, next_states, is_terminals = self.unpack_batch(states, actions, rewards, next_states, is_terminals)
        actions = actions.reshape(-1, self.env.action_dim) if self.env.has_continuous_action_space else actions.squeeze()
        rewards, states, next_states = self._check_update_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals

    def _check_update_normalize(self, rewards, states, next_states):
        if self.cfg.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
            all_states = np.concatenate([states, next_states], axis=0)
            self.state_normalizer.update(all_states)
        if self.cfg.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
            self.reward_normalizer.update(rewards)
        return rewards, states, next_states

V1 = TypeVar('OffPolicy', bound='VRLArgs')
class OffPolicy(VRL[V1]):
    def __init__(self, env, args:V1, **kwargs):
        super().__init__(env, args, **kwargs)
        self.normalizer_buffer = []

    def train(self):
        start = time.time()
        report_items = {}
        cur_s, _ = self.env.reset()
        while self.episode.sum() < self.cfg.max_epochs and self.timestep.sum() < self.cfg.max_timesteps:
            next_s, action, reward, done, info = self.step(self.env, cur_s)
            self.buffer.add((cur_s, action, reward, next_s, done))
            self.normalizer_buffer.append((cur_s, next_s, reward)) if len(self.normalizer_buffer) < 100 else self._update_normalizer()                
            self.timestep += 1
            
            if done.any():
                next_s = self.reset(done)
            cur_s = next_s
            if self.timestep.sum() > self.cfg.expl_steps:
                report_items = self._update()

                if hasattr(self, "sync_freq") and self.timestep[0] % self.cfg.sync_freq == 0 and "DQN" in self.cfg.alg_name:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                early_stop = self.monitor.timestep_report(report_items)
                if early_stop:
                    break
            
            end = time.time()
            self.training_time = (end - start)

    def _sample(self, batch_size):
        per_env_batch_size = batch_size // self.cfg.num_envs
        states, actions, rewards, next_states, is_terminals = self.buffer.sample(per_env_batch_size)
        states, actions, rewards, next_states, is_terminals = self.unpack_batch(states, actions, rewards, next_states, is_terminals)
        actions = actions.reshape(-1, self.env.action_dim) if self.env.has_continuous_action_space else actions.squeeze()
        rewards, states, next_states = self._check_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals

    def _update_normalizer(self):
        states, next_states, rewards = map(np.array, zip(*self.normalizer_buffer))
        states, next_states, rewards = self.unpack_batch(states, next_states, rewards)
        if self.cfg.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
            all_states = np.concatenate([states, next_states], axis=0)
            self.state_normalizer.update(all_states)
        if self.cfg.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
            self.reward_normalizer.update(rewards)
        self.normalizer_buffer = []

    def _check_normalize(self, rewards, states, next_states):
        if self.cfg.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.cfg.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
        return rewards, states, next_states