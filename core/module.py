import gymnasium as gym
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
from dataclasses import asdict
from abc import ABC, abstractmethod
from core.args import *
from core import noDeepLearning
from core.monitor import RLMonitor
import copy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class RL(ABC):
    def __init__(self, 
                 env, 
                 args,
                 **kwargs,
                 ):  # env_related, user_input, fixed_args
        ## env related attributes ##
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.action_space = self.env.action_space
        self.action_num =  sum(self.env.action_space.shape) if type(env.action_space) == gym.spaces.box.Box else self.env.action_space.n
        self.state_space = self.env.observation_space
        self.state_num = sum(self.env.observation_space.shape) if type(env.observation_space) == gym.spaces.box.Box else self.env.observation_space.n
        self.max_episode_steps = self.env.spec.max_episode_steps # passed when gym.make
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## hyperparameters, loading from arguments ##
        self.max_epochs:int = np.inf
        self.max_timesteps:int = np.inf
        self.reward_threshold:float = None
        self.early_stop:bool = True
        self.baseline:float = 0
        self.gamma:float = 0.99
        self.lr:float = 1e-4
        self.h_size:int = 32
        ## epoch_report related ##
        self.alg_name:str = kwargs.get("alg_name", None)
        self.model_name:str = kwargs.get("model_name", None)
        self.timestep_freq:int = None 
        self.report_freq :int = None
        self.window_size:int = 10
        ## loading asigned arguments from args ##
        default_args = asdict(Args()) if args is None else asdict(args)
        if args and hasattr(args, 'custom_args'):
            default_args.update(args.custom_args)
        default_args.update(kwargs)
        for k, v in default_args.items():
            setattr(self, k, v)
        ## logger ##
        self.logger = logging.getLogger(name=self.alg_name)
        self.monitor = RLMonitor(self)
        self.monitor._check_args()
        ## recoding arguments, variable ## 
        self.timestep = 0
        self.epoch = 0
        self.optimal_reward = 0
        self.best = None
        self.epoch_record = []
        self.timestep_record = {"timesteps": [], "rewards": []}

    @abstractmethod
    def act(self, state, mode="train"):
        raise NotImplementedError("Subclasses must implement act()")

    @abstractmethod
    def train(self):
        raise NotImplementedError("Subclasses must implement train()")

    @abstractmethod
    def _update(self):
        raise NotImplementedError("Subclasses must implement _update()")

    def timestep_evaluate(self, timestep):
        """
        Evaluate the model's performence at interval of timesteps.
        """
        self.timestep_record['timesteps'].append(timestep)
        rewards = 0

        self.timestep_record['rewards'].append(rewards)

    def evaluate(self):
        """
        Evaluate the model's performence by choosing the most likly or most valuable action.
        """
        s = self.eval_env.reset()[0]
        rewards = 0
        while True:
            a = self.act(s, mode="evaluate")
            s, reward, terminated, truncated, info = self.eval_env.step(a)
            rewards += reward
            if terminated or truncated:
                break
        return rewards
    
    def save(self, best=True):
        """
        Save the specified model's state dict.
        Args:
            model_name (str): The name of the model attribute to save. Defaults to 'policy_net';
            best (str): Whether to save the history best model, or will save the last episode's model if set to False
        """
        result_path = self.monitor._check_dir()
        save_path = os.path.join(result_path, f"{self.alg_name}_{self.env.spec.id}_h{self.h_size}.pth")
        
        assert hasattr(self, self.model_name), "Model '{self.model_name}' not found. Please ensure the model exists before saving."

        if best:
            torch.save(self.best.state_dict(), save_path)
        else:
            model = getattr(self, self.model_name)
            torch.save(model.state_dict(), save_path)
        self.logger.info(f"Model {self.alg_name}'s {self.model_name} has been saved at {save_path}, best {best}")


class VRL(RL):
    def __init__(self, env, args: VRLArgs = None, **kwargs):
        super().__init__(env=env, args=args, **kwargs)
        self.epsilon_start:float = 1.0
        self.epsilon_end:float = 0.01
        self.epsilon_decay:float = 0.002
        self.epsilon_decay_flag:bool = True
        self.noise = True if "Noisy" in self.alg_name else False

    def epsilon_greedy(self, state):
        """
        Epsilon greedy, balance eploration and usage, and has exponential decay. 
        """
        if self.epsilon_decay_flag:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.timestep *  self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_start  

        if np.random.rand() < self.epsilon:             
            return np.random.randint(self.action_num)
        else: 
            return self.act(state, mode="evaluate")  # otherwise, choose the best action based on policy


class PRL(RL):
    def __init__(self, env, args = None,**kwargs):
        super().__init__(env=env, args=args, **kwargs)
    
    def gae(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantages_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.tensor(np.array(advantages_list), dtype=torch.float).to(self.device)


# if __name__ == "__main__":
#     env = gym.make('CartPole-v1', render_mode="rgb_array")
#     agent = RL(env, reward_threshold=None)
#     print(agent._check_dir())