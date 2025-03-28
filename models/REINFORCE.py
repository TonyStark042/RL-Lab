import os
from typing import Literal
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical       
from collections import deque
from core.module import PRL, PRLArgs
from core.net import Policy_net


class REINFORCE(PRL):
    def __init__(self, env, args):       
        super().__init__(env=env, args=args, alg_name="REINFORCE", model_name="policy_net")
        self.policy_net = Policy_net(self.state_num, self.action_num, self.h_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def act(self, state, mode:Literal["train", "evaluate"] = "train"):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state).squeeze()
        if mode == "evaluate":
            action = probs.argmax()
            return action.item()
        else:
            prob_dist = Categorical(probs)
            action = prob_dist.sample()                                   
            return action.item(), prob_dist.log_prob(action)

    def train(self):
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            self.log_probs = []
            self.rewards = []
            s = self.env.reset()[0]

            while True:
                a, log_prob = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                self.timestep += 1
                self.log_probs.append(log_prob) 
                self.rewards.append(reward)
                
                if self.timestep_freq:
                    early_stop = self.monitor.timestep_report() 

                if terminated or truncated:
                    self.epoch_record.append(sum(self.rewards))
                    break

            self._update()

            if self.timestep_freq == None:
                early_stop = self.monitor.epoch_report()
                self.epoch += 1

            if early_stop:
                break

    def _update(self):
        policy_reward = torch.tensor(0.0).to(self.device)
        steps = len(self.rewards)
        returns = deque()   

        for t in range(steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)   
            returns.appendleft(self.gamma*disc_return_t + self.rewards[t])  # Dynamic Programming
        for log_prob, disc_return in zip(self.log_probs, returns):
            policy_reward += (-log_prob * (disc_return - self.baseline)).sum()       # log_prob * disc_return，but the default gradient descent direction is "-"，Adding "-" means gradient rise.
                                       
        self.optimizer.zero_grad()
        policy_reward.backward()
        self.optimizer.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    args = PRLArgs(max_epochs=1000, 
                   h_size=32, lr=0.001,
                   # report_freq=10
                   timestep_freq=100, 
                   max_timesteps=100
                   )

    agent = REINFORCE(env, args)
    agent.train()
    agent.monitor.learning_curve(mode="timestep")
    agent.save()