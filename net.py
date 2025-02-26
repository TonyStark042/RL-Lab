import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_net(nn.Module):
    def __init__(self, state_num, action_num, h_size):
        super().__init__()
        self.fc1 = nn.Linear(state_num, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy_net(nn.Module):
    def __init__(self, state_num, action_num, h_size):
        super().__init__()
        self.fc1 = nn.Linear(state_num, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)
