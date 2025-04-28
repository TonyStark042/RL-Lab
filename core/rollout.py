import numpy as np
from core.args import *
import time
from core.baseModule import PRL, VRL

class OnPolicy(PRL):
    def __init__(self, env, args=None, **kwargs):
        super().__init__(env, args, **kwargs)

    def train(self):
        start = time.time()
        reach_maxTimestep = False
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            cur_s = self.env.reset()[0]
            rewards = 0.0
            report_items = {}
            while True:
                if self.norm_obs:
                    a = self.act(self.state_normalizer(cur_s))
                else:
                    a = self.act(cur_s)
                next_s, reward, terminated, truncated, info = self.env.step(a.squeeze())
                done = terminated or truncated
                trainsition = (cur_s, a, reward, next_s, done)
                self.buffer.add(trainsition)

                cur_s = next_s
                rewards += reward                
                self.timestep += 1

                if self.buffer.is_ready(done=done, timestep=self.timestep): 
                    report_items = self._update()

                early_stop = self.monitor.timestep_report(report_items)
                reach_maxTimestep = self.timestep >= self.max_timesteps
                if early_stop or reach_maxTimestep:
                    break

                if done:
                    self.monitor.episode_evaluate()
                    self.epoch_record.append(rewards)
                    self.epoch += 1
                    break

            if early_stop or reach_maxTimestep:
                break
            
        end = time.time()
        self.training_time += (end - start)
    
    def _sample_all(self, clear=True):
        states, actions, rewards, next_states, is_terminals = self.buffer.sample_all(clear=clear)
        rewards, states, next_states = self._check_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals

class OffPolicy(VRL):
    def __init__(self, env, args=None, **kwargs):
        super().__init__(env, args, **kwargs)

    def train(self):
        start = time.time()
        reach_maxTimestep = False
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            rewards = 0
            cur_s = self.env.reset()[0]
            while True:
                if self.norm_obs:
                    a = self.act(self.state_normalizer(cur_s))
                else:
                    a = self.act(cur_s)
                next_s, reward, terminated, truncated, info = self.env.step(a)
                done = terminated or truncated
                if self.norm_obs:
                    cur_s = self.state_normalizer(cur_s)
                    next_s = self.state_normalizer(next_s)
                if self.norm_reward:
                    reward = self.reward_normalizer(reward)
                self.memory.add((cur_s, a, reward, next_s, terminated)) 

                rewards += reward
                cur_s = next_s
                report_dict = self._update()

                if hasattr(self, "sync_freq") and self.timestep % self.sync_freq == 0 and "DQN" in self.alg_name:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                early_stop = self.monitor.timestep_report(report_dict)
                self.timestep += 1
                reach_maxTimestep = self.timestep >= self.max_timesteps
                if early_stop or reach_maxTimestep:
                    break

                if done:
                    self.monitor.episode_evaluate()
                    self.epoch_record.append(rewards)
                    self.epoch += 1
                    break
            
            if "DQN" in self.alg_name and self.noise:
                self.policy_net.reset_noise()
                self.target_net.reset_noise()
            
            if early_stop or reach_maxTimestep:
                break
        end = time.time()
        self.training_time += (end - start)
    
    def _sample(self, batch_size):
        states, actions, rewards, next_states, is_terminals = self.buffer.sample(batch_size)
        rewards, states, next_states = self._check_normalize(rewards, states, next_states)
        return states, actions, rewards, next_states, is_terminals


