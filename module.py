import gymnasium as gym
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
from collections import deque
import random
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

noDeepLearning = ["Q-Learning", "Sarsa"]

@dataclass(kw_only=True)
class Args:
    alg_name:str = None
    model_name:str = None
    epochs:int = np.inf
    reward_threshold:float = None
    early_stop:bool = True
    baseline:float = 0
    gamma:float = 0.99
    lr:float = 1e-4
    h_size:int = 32
    window_size:int = 10
    custom_args: dict = field(default_factory=dict)

    def __post_init__(self):
        pass

class RL(ABC):
    def __init__(self, 
                 env, 
                 args,
                 **kwargs,
                 ):
        # env related attributes
        self.env = env
        self.action_space = self.env.action_space
        self.action_num =  sum(self.env.action_space.shape) if type(env.action_space) == gym.spaces.box.Box else self.env.action_space.n
        self.state_space = self.env.observation_space
        self.state_num = sum(self.env.observation_space.shape) if type(env.observation_space) == gym.spaces.box.Box else self.env.observation_space.n
        self.max_episode_steps = self.env.spec.max_episode_steps # passed when gym.make
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # hyperparameters
        default_args = asdict(Args()) if args is None else asdict(args)
        if args and hasattr(args, 'custom_args'):
            default_args.update(args.custom_args)
        default_args.update(kwargs)
        for k, v in default_args.items():
            setattr(self, k, v)
        self._check_args()
        # training related attributes, fixed initial value
        self.rewards_record = []
        self.optimal_reward = 0
        self.sample_count = 0
        self.best = None
    
    def _check_args(self):
        """
        Check some key argument values and print all hyperparameters.
        """

        if self.reward_threshold is None or self.reward_threshold == 0:
            self.reward_threshold = np.inf if self.env.spec.reward_threshold is None else self.env.spec.reward_threshold
        elif self.reward_threshold < 0:
            raise ValueError("reward_threshold must be a non-negative number.")
    
        print(''.join(['=']*140))
        tplt = "{:^30}\t{:^60}\t{:^30}"
        print(tplt.format("Arg","Value","Type"))
        for k,v in self.__dict__.items():
            if k == "env":
                v = self.env.spec.id
            print(tplt.format(str(k),str(v).replace(" ", ''),str(type(v))))   
        print(''.join(['=']*140)) 

        if self.max_episode_steps is None:
            print("Warning: max_episode_steps is not specified, please make sure the env must have an end.")
        if self.model_name is None and self.alg_name not in noDeepLearning:
            raise ValueError("Please specify the name of learnable policy net, which will be used in saving and updating best model")
        if self.alg_name is None:
            raise ValueError("Please specify the name of the algorithm, which will be used in saving model and learning curve")
        if self.epochs is np.inf:
            if self.early_stop:
                print("Warning: epochs is not specified, the training will continue until reward_threshold are met.")
            else:
                raise ValueError("Please specify the number of epochs for training, or set the early_stop to True.")

    @abstractmethod
    def act(self, state, mode="train"):
        raise NotImplementedError("Subclasses must implement act()")

    @abstractmethod
    def train(self):
        raise NotImplementedError("Subclasses must implement train()")
    
    @abstractmethod
    def _update(self):
        raise NotImplementedError("Subclasses must implement _update()")
    
    def report(self, epoch, **kwargs) -> bool:
        """
        Report the model reward situation in rencent <window_size> episode, and save the best model.
        Args:
            epoch: The current episode;
            early_stop: Default Fasle, if True, model will stop training if meet the reward_threshold conditions,.
        """
        if len(self.rewards_record) <= 10:
            avg_10_reward = sum(self.rewards_record) / len(self.rewards_record)
        else:
            avg_10_reward = sum(self.rewards_record[-self.window_size:])/self.window_size

        optimal_reward = self.evaluate()
        if optimal_reward >= self.optimal_reward and self.alg_name not in noDeepLearning:
            self.optimal_reward = optimal_reward
            state_dict = getattr(self, self.model_name)
            self.best = state_dict
        
        if epoch % 10 == 0:       
            print(f"Episode: {epoch}\tAverage {self.window_size}reward: {avg_10_reward:.3f}\tOptimal reward: {optimal_reward}\tHistory optimal: {self.optimal_reward}", end='\t')
            for k,v in kwargs.items():
                print(f"{k}: {v:.3f}", end='\t')
            else:
                print()

        if self.early_stop:
            if avg_10_reward >= self.reward_threshold and optimal_reward >= self.reward_threshold:
                print(f"Converged at epoch: {epoch}, final optimal reward: {optimal_reward}")
                return True
        return False
    
    def evaluate(self):
        """
        Evaluate the model's performence by choosing the most likly or most valuable action.
        """
        s = self.env.reset(seed=42)[0]
        rewards = 0
        while True:
            a = self.act(s, mode="evaluate")
            s, reward, terminated, truncated, info = self.env.step(a)
            rewards += reward
            if terminated or truncated:
                break
        return rewards   
    
    def learning_curve(self):
        """
        Plot and save the learning curve with optional moving average and visualization.
        Args:
            window_size (int): Size of the moving average window. Default is 10.
        """
        result_path = self._check_dir()
        name = f"{self.alg_name}_{self.env.spec.id}"

        episodes = range(len(self.rewards_record))
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.rewards_record, alpha=0.3, color='gray', label='Raw Rewards')

        if hasattr(self, 'reward_threshold') and self.reward_threshold not in (None, np.inf):
            plt.axhline(y=self.reward_threshold, color='r', linestyle='--', label='Reward Threshold')
            plt.legend()
        plt.axhline(y=self.optimal_reward, color='green', linestyle='--', label='Optimal Reward')
        
        if len(self.rewards_record) >= self.window_size:
            moving_avg = np.convolve(self.rewards_record, 
                                    np.ones(self.window_size)/self.window_size, 
                                    mode='valid') # only start calculating when epoch >= 10
            plt.plot(episodes[self.window_size-1:], moving_avg, 
                    color='blue', label=f'Moving Average (n={self.window_size})')
        # plt.text(0.05, 0.95, self.lr, transform=plt.gca().transAxes,  fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.title(self.env.spec.id)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()
        
        save_path = os.path.join(result_path, f"{name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)         
        print(f"Learning curve has been saved to {save_path}")
    

    def save(self, best=True):
        """
        Save the specified model's state dict.
        Args:
            model_name (str): The name of the model attribute to save. Defaults to 'policy_net';
            best (str): Whether to save the history best model, or will save the last episode's model if set to False
        """
        result_path = self._check_dir()
        save_path = os.path.join(result_path, f"{self.alg_name}_{self.env.spec.id}_h{self.h_size}.pth")
        
        assert hasattr(self, self.model_name), "Model '{self.model_name}' not found. Please ensure the model exists before saving."

        if best:
            torch.save(self.best.state_dict(), save_path)
        else:
            model = getattr(self, self.model_name)
            torch.save(model.state_dict(), save_path)
        print(f"Model {self.alg_name}'s {self.model_name} has been saved at {save_path}, best {best}")


    def _check_dir(self):
        """
        Check if there is the model's saving directory.
        """
        result_path = os.path.join("results", self.alg_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        return result_path



@dataclass(kw_only=True)
class VRLArgs(Args):
    epsilon_start:float = 1.0
    epsilon_end:float = 0.01
    epsilon_decay:float = 0.002
    epsilon_decay_flag:bool = True

class VRL(RL):
    def __init__(self, env, args: VRLArgs = None, **kwargs):
        super().__init__(env=env, args=args, **kwargs)
        self.noise = True if "Noisy" in self.alg_name else False

    def epsilon_greedy(self, state):
        """
        Epsilon greedy, balance eploration and usage, and has exponential decay. 
        """
        if self.epsilon_decay_flag:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.sample_count *  self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_start  

        if np.random.rand() < self.epsilon:             
            return np.random.randint(self.action_num)
        else: 
            return self.act(state, mode="evaluate")  # otherwise, choose the best action based on policy

    def report(self, epoch, **kwargs):
        if self.noise:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
            if super().report(epoch, weight_epsilon=self.policy_net.fc2.weight_epsilon.mean().item(), bias_epioslon=self.policy_net.fc2.bias_epsilon.mean().item(), **kwargs):
                return True
        else:
            if super().report(epoch, epsilon=self.epsilon, **kwargs):
                return True


@dataclass(kw_only=True)
class PRLArgs(Args):
    is_gae:bool = False
    lmbda:float = 0.95

class PRL(RL):
    def __init__(self, env, args = None,**kwargs):
        super().__init__(env=env, args=args, **kwargs)

    def report(self, epoch, **kwargs):
        return super().report(epoch, **kwargs)
    
    def gae(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantages_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.tensor(np.array(advantages_list), dtype=torch.float).to(self.device)


class ReplayBuffer(object):
    def __init__(self, capacity:int=10000) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self,transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size:int, sequential:bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
        
    def sample_all(self):
        return zip(*self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# if __name__ == "__main__":
#     env = gym.make('CartPole-v1', render_mode="rgb_array")
#     agent = RL(env, reward_threshold=None)
#     print(agent._check_dir())