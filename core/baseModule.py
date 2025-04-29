from gymnasium.vector import AsyncVectorEnv
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
from utils import Normalizer, make_env

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
        self.model_names:str = kwargs.get("model_names", None)
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
        ## record above arguments, must be after checking because the reward_threshold will be reset ##
        self.args = self.__dict__.copy()
        ## env related attributes ##
        self.eval_env = copy.deepcopy(self.env)
        self.action_space = self.env.action_space  # high and low are only available in continuous action space
        self.action_dim =  self.action_space.shape[0] if self.has_continuous_action_space else 1
        self.action_num = np.inf if self.has_continuous_action_space else self.action_space.n
        self.state_space = self.env.observation_space
        self.state_dim = self.state_space.shape[0] if len(self.state_space.shape) != 0 else 1
        self.state_num = self.state_space.n if hasattr(self.state_space, "n") else np.inf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## state and reward normalizer ##
        if self.norm_obs:
            if hasattr(self, "stateNormalizer"):
                self.state_normalizer = Normalizer.loading_normalizer(**self.stateNormalizer)
            else:
                self.state_normalizer = Normalizer.init_normalizer(self.env, mode="state", epochs=100)
        if self.norm_reward:
            if hasattr(self, "rewardNormalizer"):
                self.reward_normalizer = Normalizer.loading_normalizer(**self.rewardNormalizer)
            else:
                self.reward_normalizer = Normalizer.init_normalizer(self.env, mode="reward", epochs=100)
        ## parallel env ##
        if self.num_envs > 1:
            self.env = AsyncVectorEnv([make_env(self.env_name, self.max_episode_steps) for _ in range(self.num_envs)])
            self.logger.info(f"Creating {self.num_envs} parallel environments...")
        ## recoding arguments, variable ## 
        self.timestep = np.zeros(self.num_envs, dtype=np.int32) 
        self.episode = np.zeros(self.num_envs, dtype=np.int32)
        self.optimal_reward = -np.inf
        self.best = {}
        self.episode_record = np.zeros(self.num_envs, dtype=np.float32)
        self.timestep_eval = {"timesteps": [], "rewards": []}
        if self.episode_eval_freq is not None:
            self.episode_eval = {"timesteps": [], "rewards": []}
        self.training_time = 0

    @abstractmethod
    def act(self, state, mode="train"):
        raise NotImplementedError("Subclasses must implement act()")

    @abstractmethod
    def _update(self):
        raise NotImplementedError("Subclasses must implement _update()")

    def timestep_evaluate(self, timestep):
        """
        Evaluate the model's performence at interval of timesteps.
        """
        self.timestep_eval['timesteps'].append(timestep)
        rewards = 0

        self.timestep_eval['rewards'].append(rewards)
    
    def evaluate(self):
        """
        Evaluate the model's performence in training process.
        """
        results = []
        for _ in range(self.eval_epochs):
            epoch_reward = 0
            s = self.eval_env.reset()[0]
            while True:
                if self.norm_obs:
                    a = self.act(self.state_normalizer(s), deterministic=True)
                else:
                    a = self.act(s, deterministic=True)
                s, reward, terminated, truncated, info = self.eval_env.step(a.squeeze())
                epoch_reward += reward
                if terminated or truncated:
                    break
            results.append(epoch_reward)
        rewards = np.array(results)
        return rewards
    
    def test(self, save_dir=None):
        """
        Test the model's performence.
        """
        if save_dir is None:
            save_dir = self.monitor._check_dir()
        
        if self.alg_name in noDeepLearning:
            para = os.path.join(save_dir, "Q_table.npy")
            self.Q = np.load(para, allow_pickle=True)
        else:
            para = os.path.join(save_dir, "weight.pth")
            models_dict = torch.load(para)
            for net_name, state_dict in models_dict.items():
                model = getattr(self, net_name)
                model.load_state_dict(state_dict)
        self.logger.info(f"Loading model from {para}")     
        rewards = self.evaluate()
        self.logger.info(f"{self.alg_name} in {self.env_name}, Average {self.eval_epochs} reward {np.mean(rewards):.3f}, Standard deviation {np.std(rewards):.3f}")

    def save(self, best=True, save_dir=None):
        """
        Save the specified model's state dict.
        Args:
            model_names (str): The name of the model attribute to save. Defaults to 'policy_net';
            best (str): Whether to save the history best model, or will save the last episode's model if set to False
        """
        running_para = self.args
        running_para.pop("monitor")
        running_para.pop("logger")
        running_para.pop("model_names")
        running_para.pop("env")
        running_para.pop("mode")
        running_para.pop("has_continuous_action_space")
        if self.episode_eval_freq is None:
            running_para.pop("episode_eval_freq")
        if self.norm_obs:
            running_para["stateNormalizer"] = self.state_normalizer.__dict__
        if self.norm_reward:
            running_para["rewardNormalizer"] = self.reward_normalizer.__dict__

        if save_dir is None:
            save_dir = self.monitor._check_dir()
        with open(os.path.join(save_dir, 'recipe.yaml'), 'w') as f:
            yaml.dump(running_para, f)

        def _save_model(self, save_dir, best=True):
            save_dict = {}
            if self.alg_name not in noDeepLearning:
                for net_name in self.model_names:
                    if best:
                        save_dict[net_name] = self.best[net_name].state_dict()
                    else:
                        save_dict[net_name] = getattr(self, net_name).state_dict()
                save_path = os.path.join(save_dir, f"weight.pth")
                torch.save(save_dict, save_path)
                self.logger.info(f"Model {self.alg_name}'s {self.model_names} has been saved at {save_path}, best {best}")
            else:
                save_path = os.path.join(save_dir, f"Q_table.npy")
                np.save(save_path, self.Q)

        _save_model(self, save_dir, best=best)

    def _check_normalize(self, rewards, states, next_states):
        if self.norm_obs:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
            all_states = np.concatenate([states, next_states], axis=0)
            self.state_normalizer.update(all_states)
        if self.norm_reward:
            rewards = self.reward_normalizer.normalize(rewards) 
            self.reward_normalizer.update(rewards)
        return rewards, states, next_states

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

        actions = []

        if np.random.rand() < self.epsilon:
            state_num = state.reshape(-1, self.state_dim).shape[0]
            return np.random.randint(self.action_num + 1, size=state_num) # torch.randint(self.action_dim, (1,1))
        else: 
            return self.act(state, deterministic=True)  # otherwise, choose the best action based on policy


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
            advantage = self.gamma * self.gae_lambda * advantage * (1 - done) + delta
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