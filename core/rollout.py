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
            while True:
                a = self.act(cur_s)
                next_s, reward, terminated, truncated, info = self.env.step(a.squeeze())
                done = terminated or truncated
                trainsition = (cur_s, a, reward, next_s, done)
                self.buffer.add(trainsition)

                cur_s = next_s
                rewards += reward
                
                if self.buffer.is_ready(done=done, timestep=self.timestep): 
                    self._update()
                
                early_stop = self.monitor.timestep_report()
                self.timestep += 1
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


class OffPolicy(VRL):
    def __init__(self, env, args=None, **kwargs):
        super().__init__(env, args, **kwargs)

    def train(self):
        start = time.time()
        reach_maxTimestep = False
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            rewards = 0
            s = self.env.reset()[0]
            while True:
                a = self.act(s)
                next_s, reward, terminated, truncated, info = self.env.step(a)
                done = terminated or truncated
                self.memory.add((s, a, reward, next_s, terminated)) 

                rewards += reward
                s = next_s
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


