import os
from typing import Literal
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical       
from collections import deque
from module import PRL, PRLArgs
from net import Policy_net


class REINFORCE(PRL):
    def __init__(self, env, args):       
        super().__init__(env=env, args=args)
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
        for epoch in range(self.epochs):
            self.epoch_log_probs = []
            self.epoch_rewards = []
            s = self.env.reset(seed=42)[0]

            while True:
                a, log_prob = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                self.epoch_log_probs.append(log_prob) 
                self.epoch_rewards.append(reward)                              
                if terminated or truncated:
                    break
            self.rewards_record.append(sum(self.epoch_rewards))

            self._update()
            if self.report(epoch):
                break

    def _update(self):
        policy_reward = torch.tensor(0.0).to(self.device)
        steps = len(self.epoch_rewards)
        returns = deque()   

        for t in range(steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)   
            returns.appendleft(self.gamma*disc_return_t + self.epoch_rewards[t])  # Dynamic Programming
        for log_prob, disc_return in zip(self.epoch_log_probs, returns):
            policy_reward += (-log_prob * (disc_return - self.baseline)).sum()       # log_prob * disc_return，but the default gradient descent direction is "-"，Adding "-" means gradient rise.
                                       
        self.optimizer.zero_grad()
        policy_reward.backward()
        self.optimizer.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    args = PRLArgs(epochs=1000, h_size=32, alg_name="REINFORCE", model_name="policy_net", lr=0.01)

    agent = REINFORCE(env, args)
    agent.train()
    agent.learning_curve()
    agent.save()