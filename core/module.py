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
        self.max_episode_steps = self.env.spec.max_episode_steps # passed when gym.make, or use the default value
        self.has_continuous_action_space: bool = True if env.action_space.dtype in [np.float32, np.float64] else False
        self.model_name:str = kwargs.get("model_name", None)
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
        ## env related attributes ##
        self.eval_env = copy.deepcopy(env)
        self.action_space = self.env.action_space
        self.action_num =  sum(self.env.action_space.shape) if env.action_space.dtype in (np.float32, np.float64) else self.env.action_space.n
        self.state_space = self.env.observation_space
        self.state_num = sum(self.env.observation_space.shape) if env.observation_space.dtype in (np.float32, np.float64) else self.env.observation_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## recoding arguments, variable ## 
        self.timestep = 0
        self.epoch = 0
        self.optimal_reward = -np.inf
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
    
    def evaluate(self, mode="evaluate"):
        """
        Evaluate the model's performence in training process.
        """
        results = []
        # processes = min(mp.cpu_count(), self.eval_epochs)
        # with mp.Pool(processes=processes) as pool:
        # with ThreadPoolExecutor(processes) as pool: 
        #     results = list(pool.map(self.single_evaluate, range(self.eval_epochs)))
            # pool.close()
            # pool.join()
        for _ in range(self.eval_epochs):
            epoch_reward = 0
            s = self.eval_env.reset()[0]
            while True:
                a = self.act(s, mode=mode)
                s, reward, terminated, truncated, info = self.eval_env.step(a.squeeze())
                epoch_reward += reward
                if terminated or truncated:
                    break
            results.append(epoch_reward)
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

        rewards = self.evaluate(mode="test")
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
        running_para.pop("monitor")
        running_para.pop("logger")
        running_para.pop("model_name")
        running_para.pop("env")
        running_para.pop("mode")
        running_para.pop("has_continuous_action_space")

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
        if self.has_continuous_action_space:
            assert all(abs(self.action_space.low) == abs(self.action_space.high)), "Continuous action space must be symmetric"
        self.max_action = self.env.action_space.high[0] if hasattr(self.env.action_space, "high") else 1
    
    @torch.no_grad()
    def gae(self, td_delta, is_terninated):
        td_delta = td_delta.detach().numpy().flatten()
        is_terninated = is_terninated.detach().numpy().flatten()
        advantages_list = []
        advantage = 0.0
        for delta, done in zip(td_delta[::-1], is_terninated[::-1]):
            advantage = self.gamma * self.lmbda * advantage * done + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.tensor(np.array(advantages_list), dtype=torch.float, device=self.device).unsqueeze(-1)

    def adapt_action(self, a):
        if self.has_continuous_action_space:
            a = torch.clamp(a, 0, 1)
            return 2*(a-0.5)*self.max_action # limitation: maximum action expansion is consistent across all dimensions, e.g. (-2,2)
        else:
            return a
          

# if __name__ == "__main__":
#     env = gym.make('CartPole-v1', render_mode="rgb_array")
#     agent = RL(env, reward_threshold=None)
#     print(agent._check_dir())