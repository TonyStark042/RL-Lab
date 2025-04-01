import gymnasium as gym
import numpy as np
import os
import torch
from dataclasses import asdict
from abc import ABC, abstractmethod
from core.args import *
from core.monitor import RLMonitor
from core import noDeepLearning
import copy
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class RL(ABC):
    def __init__(self, 
                 env, 
                 args,
                 **kwargs,
                 ): 
        ## hyperparameters, loading from arguments ##
        self.env = env
        self.env_name = env.spec.id
        self.max_episode_steps = self.env.spec.max_episode_steps # passed when gym.make
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
        self.eval_epochs:int = 10
        ## loading asigned arguments from args ##
        default_args = asdict(args)
        default_args.pop("max_episode_steps")
        default_args.update(kwargs)
        for k, v in default_args.items():
            setattr(self, k, v)

        ## logger ##
        self.logger = logging.getLogger(name=self.alg_name)
        self.monitor = RLMonitor(self)
        self.monitor._check_args()
        # record above arguments, must be after checking, the reward_threshold will be reset
        self.args = self.__dict__.copy()  
        self.args.pop("monitor")
        self.args.pop("logger")
        self.args.pop("model_name")
        self.args.pop("env")
        self.args.pop("mode")

        ## env related attributes ##
        self.eval_env = copy.deepcopy(env)
        self.action_space = self.env.action_space
        self.action_num =  sum(self.env.action_space.shape) if type(env.action_space) == gym.spaces.box.Box else self.env.action_space.n
        self.state_space = self.env.observation_space
        self.state_num = sum(self.env.observation_space.shape) if type(env.observation_space) == gym.spaces.box.Box else self.env.observation_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## recoding arguments, variable ## 
        self.timestep = 0
        self.epoch = 0
        self.optimal_reward = 0
        self.best = None
        self.epoch_record = []
        self.timestep_record = {"timesteps": [], "rewards": []}
        self.training_time = 0

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

    def single_evaluate(self):
        s = self.eval_env.reset()[0]
        epoch_reward = 0
        while True:
            a = self.act(s, mode="evaluate")
            s, reward, terminated, truncated, info = self.eval_env.step(a)
            epoch_reward += reward
            if terminated or truncated:
                break
        return epoch_reward
    
    def evaluate(self):
        """
        Evaluate the model's performence by choosing the most likly or most valuable action.
        """
        results = []
        # processes = min(mp.cpu_count(), self.eval_epochs)
        # with mp.Pool(processes=processes) as pool:
        # with ThreadPoolExecutor(processes) as pool: 
        #     results = list(pool.map(self.single_evaluate, range(self.eval_epochs)))
            # pool.close()
            # pool.join()
        for _ in range(self.eval_epochs):
            results.append(self.single_evaluate())
        rewards = np.array(results)
        return rewards
    
    def test(self):
        """
        Test the model's performence.
        """
        result_dir = self.monitor._check_dir()
        
        if self.alg_name in noDeepLearning:
            para = os.path.join(result_dir, "Q_table.npy")
            self.Q = np.load(para, allow_pickle=True)
        else:
            para = os.path.join(result_dir, "weight.pth")
            model = getattr(self, self.model_name)
            model.load_state_dict(torch.load(para))
        self.logger.info(f"Loading model from {para}")

        rewards = self.evaluate()
        self.logger.info(f"{self.alg_name} in {self.env_name}, Average {self.eval_epochs} reward {np.mean(rewards):.3f}, Standard deviation {np.std(rewards):.3f}")

    def save(self, best=True):
        """
        Save the specified model's state dict.
        Args:
            model_name (str): The name of the model attribute to save. Defaults to 'policy_net';
            best (str): Whether to save the history best model, or will save the last episode's model if set to False
        """
        result_path = self.monitor._check_dir()        

        running_para = self.args
        if self.alg_name in noDeepLearning:
            running_para.pop("lr")
            running_para.pop("h_size")
        elif "PPO" in self.alg_name or "A2C" in self.alg_name:
            running_para.pop("lr")
        elif "Noisy" not in self.alg_name and "DQN" in self.alg_name:
            running_para.pop("std_init")

        with open(os.path.join(result_path, 'recipe.yaml'), 'w') as f:
            yaml.dump(running_para, f)

        if self.alg_name not in noDeepLearning:
            save_path = os.path.join(result_path, f"weight.pth")
            if best:
                torch.save(self.best.state_dict(), save_path)
            else:
                model = getattr(self, self.model_name)
                torch.save(model.state_dict(), save_path)
            self.logger.info(f"Model {self.alg_name}'s {self.model_name} has been saved at {save_path}, best {best}")
        else:
            save_path = os.path.join(result_path, f"Q_table.npy")
            np.save(save_path, self.Q)




class VRL(RL):
    def __init__(self, env, args= None, **kwargs):
        super().__init__(env=env, args=args, **kwargs)
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