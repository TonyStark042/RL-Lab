from typing import Literal
from module import VRL, ReplayBuffer, VRLArgs
import os
from net import Q_net, Dueling_Q_net
from torch import optim
import torch
import numpy as np
from torch import nn
import gymnasium as gym
from argparse import ArgumentParser


class DQN(VRL):
    def __init__(self, env, args:VRLArgs=None):
        super().__init__(env, args=args)
        if "Dueling" in self.alg_name:
            self.policy_net = Dueling_Q_net(self.state_num, self.action_num, self.h_size, self.noise, self.std_init).to(self.device)
            self.target_net = Dueling_Q_net(self.state_num, self.action_num, self.h_size, self.noise, self.std_init).to(self.device)
        else:
            self.policy_net = Q_net(self.state_num, self.action_num, self.h_size, self.noise).to(self.device)
            self.target_net = Q_net(self.state_num, self.action_num, self.h_size, self.noise).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayBuffer(self.memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) # only policy_net need optimizer
    
    @torch.no_grad() # Based on Q value to select action, no need to calculate gradient
    def act(self, state, mode:Literal["train", "evaluate"]="train"):
        self.sample_count += 1
        if mode == "evaluate":
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            action = q_values.argmax().item()
        else:
            if self.noise == True:
                state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
                action = self.policy_net(state).argmax().item()
            else:
                action = self.epsilon_greedy(state)
        return action

    def train(self):
        for epoch in range(self.epochs):
            rewards = []
            s = self.env.reset(seed=42)[0]
            while True:
                a = self.act(s)
                next_s, reward, terminated, truncated, info = self.env.step(a)
                rewards.append(reward)
                self.memory.add((s, a, reward, next_s, terminated)) 
                s = next_s
                self._update()
                if self.sample_count % self.sync_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())                   
                if terminated or truncated:
                    break
            self.rewards_record.append(sum(rewards))
            
            if self.report(epoch):
                break

    def _update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)  
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)  
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.float32(dones), device=self.device)

        Q_values = self.policy_net(states).gather(dim=1, index=actions)
        
        if "Double" in self.alg_name:
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True) # Q_net select next action, but evaluated by target_net
            next_Q = self.target_net(next_states).gather(1, next_actions).squeeze()
        else:
            next_Q = self.target_net(next_states).max(dim=1)[0].detach()            
        
        targets = rewards + self.gamma * next_Q * (1 - dones)
        loss = nn.MSELoss()(Q_values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def test(self, save_dir):
        self.policy_net.load_state_dict(torch.load(save_dir))
        rewards = []
        s = self.env.reset(seed=42)[0]
        while True:
            a = self.act(s, mode="evaluate")
            next_s, reward, terminated, truncated, info = self.env.step(a)
            rewards.append(reward)
            s = next_s                 
            if terminated or truncated:
                break
        print(f"Test reward: {sum(rewards)}")
        

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array", max_episode_steps=500)
    parser = ArgumentParser(description="DQN Settings")
    parser.add_argument("--alg_name", 
                        type=str, 
                        choices=["DQN", "Double", "Dueling", "DoubleDuelingDQN", "NoisyDQN"],
                        default="DQN")
    parser_args = parser.parse_args()


    # alg_name="DQN",
    # alg_name="DoubleDQN",
    # alg_name="DoubleDuelingDQN"
    # alg_name="NoisyDQN"
    alg_name = "DoubleDuelingNoisyDQN"
    args = VRLArgs( epochs=1000,
                    h_size=64, 
                    alg_name=alg_name, 
                    model_name="policy_net",
                    custom_args={
                                "sync_steps":64, 
                                "batch_size":32, 
                                "memory_size":6000,
                                "std_init":0.4,
                                })
    
    agent = DQN(env, args=args)
    agent.train()
    # agent.learning_curve()
    # agent.save()
    save_dir = f"results/{alg_name}/{alg_name}_{env.spec.id}_h{agent.h_size}.pth"
    agent.test(save_dir=save_dir)