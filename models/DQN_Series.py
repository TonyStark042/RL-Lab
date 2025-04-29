from typing import Literal
from core.baseModule import VRL
from core.net import Q_net, Dueling_Q_net
from torch import optim
import torch
import numpy as np
from torch import nn
from core.args import VRLArgs
from core.buffer import ReplayBuffer
from core.rollout import OffPolicy

class DQN(OffPolicy):
    def __init__(self, env, args:VRLArgs=None):
        super().__init__(env, args=args, model_names=["policy_net"],)
        if "Dueling" in self.alg_name:
            self.policy_net = Dueling_Q_net(self.state_dim, self.action_num, self.h_size, self.noise, self.std_init).to(self.device)
            self.target_net = Dueling_Q_net(self.state_dim, self.action_num, self.h_size, self.noise, self.std_init).to(self.device)
        else:
            self.policy_net = Q_net(self.state_dim, self.action_num, self.h_size, self.noise).to(self.device)
            self.target_net = Q_net(self.state_dim, self.action_num, self.h_size, self.noise).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.buffer = ReplayBuffer(self.memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) # only policy_net need optimizer
    
    @torch.no_grad() # Based on Q value to select action, no need to calculate gradient
    def act(self, state, deterministic=False):
        if deterministic:
            state = torch.tensor(state, device=self.device, dtype=torch.float).reshape(-1, self.state_dim)
            action = self.policy_net(state).argmax(-1).cpu().numpy()
        else:
            if self.noise:
                state = torch.tensor(state, device=self.device, dtype=torch.float).reshape(-1, self.state_dim)
                action = self.policy_net(state).argmax(-1).cpu().numpy()
            else:
                action = self.epsilon_greedy(state)
        return action

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return {}
        states, actions, rewards, next_states, dones = self._sample(self.batch_size)
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float).view(self.batch_size, -1)
        actions = torch.tensor(np.array(actions), device=self.device).view(self.batch_size, -1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)  
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float).view(self.batch_size, -1)
        dones = torch.tensor(np.float32(dones), device=self.device)

        Q_values = self.policy_net(states).gather(dim=1, index=actions).view(self.batch_size, -1)
        
        if "Double" in self.alg_name:
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True) # Q_net select next action, but evaluated by target_net
            next_Q = self.target_net(next_states).gather(1, next_actions).squeeze()
        else:
            next_Q = self.target_net(next_states).max(dim=1)[0].detach() # normal DQN has overestimation bias due to always select <max>.   
        
        targets = rewards + self.gamma * next_Q * (1 - dones)
        loss = nn.MSELoss()(Q_values, targets.view(self.batch_size, -1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.report_item(loss=loss.item())

    def report_item(self, **kwargs):
        report_dict = kwargs
        if self.noise:
            return report_dict
        else:
            report_dict["epsilon"] = self.epsilon
            return report_dict