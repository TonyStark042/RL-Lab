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
            if self.norm_obs:
                a = self.act(self.state_normalizer(cur_s))
            else:
                a = self.act(cur_s)
            actual_a = self._check_action_dim(a)
            next_s, reward, termimations, truncations, info = self.env.step(actual_a)
            done = termimations | truncations if isinstance(termimations, np.ndarray) else np.array(termimations or truncations)
            self.buffer.add((cur_s, a, reward, next_s, done))
            self.timestep += 1
            
            if done.any():
                reset_mask = {"reset_mask":done}
                next_s = self.env.reset(options=reset_mask)[0]
                self.episode[done] += 1
                self.monitor.episode_evaluate()
            cur_s = next_s

            if self.buffer.is_ready(done=done, timestep=self.timestep[0]): 
                report_items = self._update()

            early_stop = self.monitor.timestep_report(report_items)
            if early_stop:
                    break
            
        end = time.time()
        self.training_time += (end - start)

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
        rewards, states, next_states = self._check_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals

class OffPolicy(VRL):
    def __init__(self, env, args=None, **kwargs):
        super().__init__(env, args, **kwargs)

    def train(self):
        start = time.time()
        report_items = {}
        cur_s, _ = self.env.reset()
        while self.episode.sum() < self.max_epochs and self.timestep.sum() < self.max_timesteps:
            if self.norm_obs:
                a = self.act(self.state_normalizer(cur_s))
            else:
                a = self.act(cur_s)
            actual_a = self._check_action_dim(a)
            next_s, reward, termimations, truncations, info = self.env.step(actual_a)
            done = termimations | truncations if isinstance(termimations, np.ndarray) else termimations or truncations
            self.buffer.add((cur_s, a, reward, next_s, done))
            self.timestep += 1
            
            if np.array(done).any():
                reset_mask = {"reset_mask":done}
                next_s = self.env.reset(options=reset_mask)[0]
                self.episode[done] += 1
                self.monitor.episode_evaluate()
                if "DQN" in self.alg_name and self.noise:
                    self.policy_net.reset_noise()
                    self.target_net.reset_noise()
            cur_s = next_s

            report_items = self._update()

            if hasattr(self, "sync_freq") and self.timestep[0] % self.sync_freq == 0 and "DQN" in self.alg_name:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            early_stop = self.monitor.timestep_report(report_items)
            if early_stop:
                    break
            
        end = time.time()
        self.training_time += (end - start)

    def _sample(self, batch_size):
        batch_size = batch_size // self.num_envs
        states, actions, rewards, next_states, is_terminals = self.buffer.sample(batch_size)
        if self.num_envs > 1:
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


