import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical, MultivariateNormal, Normal
from core.env import WrappedEnv

class Q_net(nn.Module):
    def __init__(self, env:WrappedEnv, h_size, noise=False, std_init=0.1):
        super().__init__()
        self.noise = noise
        self.fc1 = nn.Linear(env.state_dim, h_size)
        if self.noise == True:
            self.fc2 = NoisyLinear(h_size, h_size*2, std_init=std_init)
            self.fc3 = NoisyLinear(h_size*2, env.action_num, std_init=std_init)
        else:
            self.fc2 = nn.Linear(h_size, h_size*2)
            self.fc3 = nn.Linear(h_size*2, env.action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # To estimate the Q value of each action
        return x
    
    def reset_noise(self):
        if self.noise == False:
            pass
        else:
            self.fc2.reset_noise()
            self.fc3.reset_noise()

class Dueling_Q_net(nn.Module):
    def __init__(self, env:WrappedEnv, h_size, noise=False, std_init=0.1):
        super().__init__()
        self.noise = noise
        self.fc1 = nn.Linear(env.state_dim, h_size)
        if self.noise == True:
            self.fc2 = NoisyLinear(h_size, h_size*2, std_init=std_init)
            self.fc3 = NoisyLinear(h_size*2, 1, std_init=std_init)
            self.fc4 = NoisyLinear(h_size*2, env.action_num, std_init=std_init)
        else:
            self.fc2 = nn.Linear(h_size, h_size*2)
            self.fc3 = nn.Linear(h_size*2, 1)
            self.fc4 = nn.Linear(h_size*2, env.action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        a = self.fc4(x)
        return v + a - a.mean(dim=1, keepdim=True)  # Q = V(s) + A(s, a)

    def reset_noise(self):
        if self.noise == False:
            pass
        else:
            self.fc2.reset_noise()
            self.fc3.reset_noise()
            self.fc4.reset_noise()

class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.2):
        super(NoisyLinear, self).__init__()
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.std_init   = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(output_dim, input_dim)) # multi the transpose, so output_dim is the first dim
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))
        
        self.init_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    # initialize the mean and std of weight and bias, sampled from uniform distribution, learnable.
    def init_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class SAC_PolicyNet(nn.Module): 
    def __init__(self, env:WrappedEnv, h_size):
        super().__init__()
        self.action_dim = env.action_dim
        self.fc1 = nn.Linear(env.state_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.mu = nn.Linear(h_size, env.action_dim)
        self.logstd = nn.Linear(h_size, env.action_dim)
        self.max_logstd = 2
        self.min_logstd = -20

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        logstd = torch.clamp(logstd, min=self.min_logstd, max=self.max_logstd)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        return dist

class Policy_net(nn.Module):
    def __init__(self, env:WrappedEnv, h_size):
        super().__init__()
        self.has_continuous_action_space = env.has_continuous_action_space
        self.fc1 = nn.Linear(env.state_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        if self.has_continuous_action_space:
            self.logstd = nn.Parameter(torch.zeros(env.action_dim))
            self.mu = nn.Linear(h_size*2, env.action_dim)
        else:
            self.fc3 = nn.Linear(h_size*2, env.action_num)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))        
        if self.has_continuous_action_space:
            mu = self.mu(x)
            logstd = self.logstd.expand_as(mu)
            std = torch.exp(logstd)
            dist = Normal(mu, std)
        else:
            probs = F.softmax(self.fc3(x), dim=1)
            dist = Categorical(probs) # support probs dim is greater than 1
        return dist

class Critic_Vnet(nn.Module):   # To estimate the value of state
    def __init__(self, env:WrappedEnv, h_size):
        super().__init__()
        self.fc1 = nn.Linear(env.state_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic(nn.Module):
    def __init__(self, env:WrappedEnv, h_size):
        super().__init__()
        self.critic = Critic_Vnet(env, h_size)
        self.actor = Policy_net(env, h_size)

    def forward(self, x):
        value = self.critic(x)
        dist = self.actor(x)
        return dist, value

class Determin_PolicyNet(nn.Module): # It is not recommended to be used in discrete action space, so only support continuous.  
    def __init__(self, env:WrappedEnv, h_size):
        super().__init__()
        self.action_dim = env.action_dim
        self.fc1 = nn.Linear(env.state_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, env.action_dim)
        # Initialize all network weights and biases with small values
        self.apply(weight_init)
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
    
class Critic_Qnet(nn.Module):  # To estimate the value of state-action pair
    def __init__(self, env:WrappedEnv, h_size):
        super().__init__()
        self.fc1 = nn.Linear(env.state_dim + env.action_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, 1)
        self.apply(weight_init)
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)