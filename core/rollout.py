import numpy as np
from core.args import *
import time
from core.baseModule import PRL, VRL

class OnPolicy(PRL):
    def __init__(self, env, args=None, **kwargs):
        super().__init__(env, args, **kwargs)
    
    def train(self):
        start = time.time()
        report_items = {}
        cur_s, _ = self.env.reset()
        while self.episode.sum() < self.max_epochs and self.timestep.sum() < self.max_timesteps:
            next_s, action, reward, done, info = self.step(self.env, cur_s)
            self.buffer.add((cur_s, action, reward, next_s, done))
            self.timestep += 1
            
            if done.any():
                next_s = self.reset(done)
            cur_s = next_s

            if self.buffer.is_ready(done=done, timestep=self.timestep[0]): 
                report_items = self._update()

            early_stop = self.monitor.timestep_report(report_items)
            if early_stop:
                    break
            
            end = time.time()
            self.training_time = (end - start)

    def _sample_all(self, clear=True):
        states, actions, rewards, next_states, is_terminals = self.buffer.sample_all(clear=clear)
        if self.num_envs > 1:
            states = np.concat(list(zip(*states)), axis=0)
            actions = np.concat(list(zip(*actions)), axis=0)
            rewards = np.concat(list(zip(*rewards)), axis=0)
            next_states = np.concat(list(zip(*next_states)), axis=0)
            is_terminals = list(is_terminals)
            is_terminals[-1] = np.array([True] * self.num_envs)
            is_terminals = np.concat(list(zip(*is_terminals)), axis=0)
        actions = actions.reshape(-1, self.action_dim) if self.has_continuous_action_space else actions.squeeze()
        rewards, states, next_states = self._check_update_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals

    def _check_update_normalize(self, rewards, states, next_states):
        if self.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
            all_states = np.concatenate([states, next_states], axis=0)
            self.state_normalizer.update(all_states)
        if self.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
            self.reward_normalizer.update(rewards)
        return rewards, states, next_states

class OffPolicy(VRL):
    def __init__(self, env, args=None, **kwargs):
        super().__init__(env, args, **kwargs)
        self.normalizer_buffer = []

    def train(self):
        start = time.time()
        report_items = {}
        cur_s, _ = self.env.reset()
        while self.episode.sum() < self.max_epochs and self.timestep.sum() < self.max_timesteps:
            next_s, action, reward, done, info = self.step(self.env, cur_s)
            self.buffer.add((cur_s, action, reward, next_s, done))
            self.normalizer_buffer.append((cur_s, next_s, reward)) if len(self.normalizer_buffer) < 100 else self._update_normalizer()                
            self.timestep += 1
            
            if done.any():
                next_s = self.reset(done)
            cur_s = next_s
            if self.timestep.sum() > self.expl_steps:
                report_items = self._update()

                if hasattr(self, "sync_freq") and self.timestep[0] % self.sync_freq == 0 and "DQN" in self.alg_name:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                early_stop = self.monitor.timestep_report(report_items)
                if early_stop:
                    break
            
            end = time.time()
            self.training_time = (end - start)

    def _sample(self, batch_size):
        per_env_batch_size = batch_size // self.num_envs
        states, actions, rewards, next_states, is_terminals = self.buffer.sample(per_env_batch_size)
        if self.num_envs > 1: # however, for off policy, we usually do not use multiple envs
            states = np.concat(list(zip(*states)), axis=0)
            actions = np.concat(list(zip(*actions)), axis=0)
            rewards = np.concat(list(zip(*rewards)), axis=0)
            next_states = np.concat(list(zip(*next_states)), axis=0)
            is_terminals = list(is_terminals)
            is_terminals[-1] = np.array([True] * self.num_envs)
            is_terminals = np.concat(list(zip(*is_terminals)), axis=0)
        actions = actions.reshape(-1, self.action_dim) if self.has_continuous_action_space else actions.squeeze()
        rewards, states, next_states = self._check_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals

    def _update_normalizer(self):
        states, next_states, rewards = map(np.array, zip(*self.normalizer_buffer))
        if self.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
            all_states = np.concatenate([states, next_states], axis=0)
            self.state_normalizer.update(all_states)
        if self.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
            self.reward_normalizer.update(rewards)
        self.normalizer_buffer = []

    def _check_normalize(self, rewards, states, next_states):
        if self.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
        return rewards, states, next_states