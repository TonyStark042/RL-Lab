import time
import gymnasium as gym
from typing import Literal, Optional
import numpy as np
from core.module import VRL
from core.args import VRLArgs


class Q_Learning(VRL):
    def __init__(self, env, args):
        super().__init__(env, args=args)
        self.Q = np.zeros((self.state_dim, self.action_dim))
    
    def act(self, state, mode:Literal["train", "evaluate", "test"]="train"):
        if mode == "train":
            a = self.epsilon_greedy(state)
        else:
            a = np.argmax(self.Q[state])
        return a

    def train(self):
        start = time.time()
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            cur_s, _ = self.env.reset()
            reach_maxTimestep = False
            epoch_reward = 0
            if self.alg_name == 'Q_Learning':
                while True:
                    self.timestep += 1
                    a = self.act(cur_s)                      # behavior sugg，based on epsillon_greedy to choose the action.
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    epoch_reward += reward
                    self._update(cur_s, a, next_s, None, reward) # target sugg, using the best Q of s' to update
                    cur_s = next_s
                    if self.train_mode == "timestep":
                        early_stop = self.monitor.timestep_report()
                        reach_maxTimestep = self.timestep >= self.max_timesteps
                        if early_stop or reach_maxTimestep:
                            break
                        
                    if terminated or truncated:
                        self.epoch_record.append(epoch_reward)
                        break
            else:
                a = self.act(cur_s)  
                while True:
                    self.timestep += 1
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    epoch_reward += reward
                    next_a = self.act(next_s)
                    self._update(cur_s, a, next_s, next_a, reward)    # performing TD by the actual action in s'
                    cur_s = next_s
                    a = next_a

                    if self.train_mode == "timestep":
                        early_stop = self.monitor.timestep_report()
                        reach_maxTimestep = self.timestep >= self.max_timesteps
                        if early_stop or reach_maxTimestep:
                            break

                    if terminated or truncated:
                        self.epoch_record.append(epoch_reward)
                        break
                    
            self.epoch += 1
            if self.train_mode == "episode":
                early_stop = self.monitor.epoch_report()

            if early_stop or reach_maxTimestep:
                break 
        end = time.time()
        self.training_time += (end - start)

    def _update(self, s, a, next_s, next_a:Optional[None], r):  
        if next_a == None:
            Q_target = r + self.gamma * np.max(self.Q[next_s])             
        else:
            Q_target = r + self.gamma * self.Q[next_s][next_a]                
        self.Q[s][a] = self.Q[s][a] + self.lr * (Q_target - self.Q[s][a])