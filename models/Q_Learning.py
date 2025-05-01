import time
from typing import Literal, Optional
import numpy as np
from core.baseModule import VRL


class Q_Learning(VRL):
    def __init__(self, env, args):
        super().__init__(env, args=args)
        self.Q = np.zeros((self.state_num, self.action_num))
    
    def act(self, state, deterministic=False):
        if deterministic:
            a = np.argmax(self.Q[state])
        else:
            a = self.epsilon_greedy(state)
        return a

    def train(self):
        start = time.time()
        while self.episode.sum() < self.max_epochs and self.timestep.sum() < self.max_timesteps:
            cur_s, _ = self.env.reset()
            reach_maxTimestep = False
            a = self.act(cur_s).item()
            while True:
                if self.alg_name == "Q_Learning":
                    a = self.act(cur_s).item()
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    report_item = self._update(cur_s, a, next_s, None, reward)
                elif self.alg_name == "Sarsa":
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    next_a = self.act(next_s).item()
                    report_item = self._update(cur_s, a, next_s, next_a, reward)
                    a = next_a
                cur_s = next_s
                done = terminated or truncated

                early_stop = self.monitor.timestep_report(report_item)
                self.timestep += 1
                reach_maxTimestep = self.timestep >= self.max_timesteps
                if early_stop or reach_maxTimestep:
                    break
                    
                if done:
                    self.monitor.episode_evaluate()
                    self.episode += 1
                    break

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
        
        return {"epsilon":self.epsilon.item()}