from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym
import numpy as np
import os
import torch
from abc import ABC, abstractmethod
from core.monitor import RLMonitor
from core import noDeepLearning
import logging
import yaml
from core.utils import Normalizer
from core.args import BasicArgs, VRLArgs, PRLArgs
from core.env import WrappedEnv
from typing import Generic, TypeVar, Optional, Literal

A = TypeVar('Args', bound='BasicArgs')
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class RL(ABC, Generic[A]):
    def __init__(self, env:WrappedEnv, args: A, **kwargs):
        # 1. Set up configuration and environment 
        self.cfg = args
        self.env = WrappedEnv(env)
        self.eval_env = SyncVectorEnv([lambda: gym.make(self.cfg.env_name, max_episode_steps=self.cfg.max_episode_steps)])
        self._correct_cfg(env)
        
        # 2. Initialize training state
        self.model_names = kwargs.get("model_names", None)
        self.timestep = np.zeros(self.cfg.num_envs, dtype=np.int32)
        self.episode = np.zeros(self.cfg.num_envs, dtype=np.int32)
        self.optimal_reward = -np.inf
        self.best = dict()
        self.timestep_eval = {"timesteps": [], "rewards": []}
        self.episode_eval = {"episodes": [], "rewards": []} if self.cfg.episode_eval_freq is not None else None            
        self.training_time = 0        

        # 3. Set up utilities (logging, device, etc)
        self.logger = logging.getLogger(name=self.cfg.alg_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = RLMonitor.check_args(self)
        
        # 4. Initialize normalizers
        self.state_normalizer: Optional[Normalizer] = None
        self.reward_normalizer: Optional[Normalizer] = None
        self._setup_normalizers()
    
    def _correct_cfg(self, env):
        spec = env.get_attr("spec")[0]
        self.cfg.max_episode_steps = spec.max_episode_steps
        if self.cfg.reward_threshold is None or self.cfg.reward_threshold == 0:
            self.cfg.reward_threshold = np.inf if spec.reward_threshold is None else spec.reward_threshold
        if self.cfg.save_dir is None:
            self.cfg.save_dir = os.path.join("results", self.cfg.alg_name, f"{self.cfg.env_name}")

    def _setup_normalizers(self):
        """Set up state and reward normalizers if needed"""
        save_dir = self.monitor._check_dir()
        if self.cfg.norm_obs:
            if "stateNormalizer.npz" in os.listdir(save_dir):
                save_path = os.path.join(save_dir, 'stateNormalizer.npz')
                stateNormalizer = np.load(save_path)
                self.state_normalizer = Normalizer.loading_normalizer(**stateNormalizer)
            else:
                self.state_normalizer = Normalizer.init_normalizer(self.eval_env, mode="state", epochs=100)
        if self.cfg.norm_reward:
            if "rewardNormalizer.npz" in os.listdir(self.cfg.save_dir):
                save_path = os.path.join(save_dir, 'rewardNormalizer.npz')
                rewardNormalizer = np.load(save_path)
                self.reward_normalizer = Normalizer.loading_normalizer(**rewardNormalizer)
            else:
                self.reward_normalizer = Normalizer.init_normalizer(self.eval_env, mode="reward", epochs=100)
    
    def _setup_parallel_envs(self):
        """Set up parallel environments if needed"""
        if self.cfg.num_envs > 1:
            self.logger.info(f"Creating {self.cfg.num_envs} parallel environments...")
        env = AsyncVectorEnv([
            [lambda: gym.make(self.cfg.env_name, max_episode_steps=self.cfg.max_episode_steps) for _ in range(self.cfg.num_envs)]
        ])
        return env
    
    @abstractmethod
    def act(self, state, deterministic=False):
        raise NotImplementedError("Subclasses must implement act()")

    @abstractmethod
    def _update(self):
        raise NotImplementedError("Subclasses must implement _update()")

    def step(self, env, s , deterministic=False, mode:Literal["train", "test"]="train"):
        if hasattr(self.cfg, "expl_steps") and self.timestep.sum() < self.cfg.expl_steps and mode != "test":
            a = self.env.action_space.sample().reshape(-1, self.env.action_dim)
        elif self.cfg.norm_obs:
            a = self.act(self.state_normalizer(s), deterministic=deterministic)
        else:
            a = self.act(s, deterministic=deterministic)
        actual_a = a.reshape(-1, self.env.action_dim) if self.env.has_continuous_action_space else a.reshape(self.env.action_dim, )
        s, reward, terminated, truncated, info = env.step(actual_a)
        done = terminated | truncated if isinstance(terminated, np.ndarray) else np.array(terminated or truncated)
        return s, a, reward, done, info
    
    def reset(self, done):
        reset_mask = {"reset_mask":done}
        next_s = self.env.reset(options=reset_mask)[0]
        self.episode[done] += 1
        self.monitor.episode_evaluate()
        if "DQN" in self.cfg.alg_name and self.noise:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        return next_s

    def evaluate(self):
        """
        Evaluate the model's performence in training process.
        """
        results = []
        for _ in range(self.cfg.eval_epochs):
            epoch_reward = 0
            s = self.eval_env.reset()[0]
            while True:
                s, a, reward, done, info = self.step(self.eval_env, s, deterministic=True)
                epoch_reward += reward.item()
                if done:
                    break
            results.append(epoch_reward)
        rewards = np.array(results)
        return rewards
    
    def save(self, best=True):
        """
        Save the specified model's state dict.
        Args:
            model_names (str): The name of the model attribute to save. Defaults to 'policy_net';
            best (str): Whether to save the history best model, or will save the last episode's model if set to False
        """
        running_para = vars(self.cfg)
        save_dir = self.monitor._check_dir()
        with open(os.path.join(save_dir, 'recipe.yaml'), 'w') as f:
            yaml.dump(running_para, f)

        if self.cfg.norm_obs:
            stateNormalizer = dict()
            stateNormalizer["mean"] = self.state_normalizer.mean
            stateNormalizer["var"] = self.state_normalizer.var
            stateNormalizer["count"] = self.state_normalizer.count
            save_path = os.path.join(save_dir, 'stateNormalizer.npz')
            np.savez(save_path, **stateNormalizer)
        if self.cfg.norm_reward:
            rewardNormalizer = dict()
            rewardNormalizer["mean"] = self.reward_normalizer.mean
            rewardNormalizer["var"] = self.reward_normalizer.var
            rewardNormalizer["count"] = self.reward_normalizer.count
            save_path = os.path.join(save_dir, 'rewardNormalizer.npz')
            np.savez(save_path, **rewardNormalizer)

        def _save_model(best=True):
            save_dict = {}
            if self.cfg.alg_name not in noDeepLearning:
                for net_name in self.model_names:
                    if best:
                        save_dict[net_name] = self.best[net_name].state_dict()
                    else:
                        save_dict[net_name] = getattr(self, net_name).state_dict()
                save_path = os.path.join(save_dir, f"weight.pth")
                torch.save(save_dict, save_path)
                self.logger.info(f"Model {self.cfg.alg_name}'s {self.model_names} has been saved at {save_path}, best {best}")
            else:
                save_path = os.path.join(save_dir, f"Q_table.npy")
                np.save(save_path, self.Q)

        def _save_eval():
            if self.cfg.episode_eval_freq is not None and len(self.episode_eval["episodes"]) != 0:
                save_path = os.path.join(save_dir, 'episode_eval.npz')
                self.episode_eval["episodes"] = np.array(self.episode_eval["episodes"])
                self.episode_eval["rewards"] = np.array(self.episode_eval["rewards"])
                np.savez(save_path, **self.episode_eval)
            if len(self.timestep_eval["timesteps"]) != 0:
                save_path = os.path.join(save_dir, 'timestep_eval.npz')
                self.timestep_eval["timesteps"] = np.array(self.timestep_eval["timesteps"])
                self.timestep_eval["rewards"] = np.array(self.timestep_eval["rewards"])
                np.savez(save_path, **self.timestep_eval)

        _save_model(best=best)
        _save_eval()

    def a2a(self, action):
        """Convert action to the action space of the environment."""
        if self.env.has_continuous_action_space:
            action = action*self.env.max_action
        return action

    @staticmethod
    def unpack_batch(*values):
        """Unpack the batch data from the buffer."""
        unpacked_values = []
        for value in values:
            value = np.concat(list(zip(*value)), axis=0)
            unpacked_values.append(value)
        return unpacked_values

V = TypeVar('ValueArgs', bound='VRLArgs')
class VRL(RL[V]):
    def __init__(self, env:WrappedEnv, args:V, **kwargs):
        super().__init__(env=env, args=args, **kwargs)
        self.noise = True if "Noisy" in self.cfg.alg_name else False

    def epsilon_greedy(self, state):
        """
        Epsilon greedy, balance eploration and usage, and has exponential decay. 
        """
        if self.cfg.epsilon_decay_flag:
            self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * np.exp(-1. * self.timestep *  self.cfg.epsilon_decay)
        else:
            self.epsilon = np.array(self.cfg.epsilon_start)

        if np.random.rand() < self.epsilon:
            state_num = state.reshape(-1, self.env.state_dim).shape[0] if type(state) == np.ndarray else 1
            return np.random.randint(self.env.action_num, size=state_num) # torch.randint(self.env.action_dim, (1,1))
        else: 
            return self.act(state, deterministic=True)  # otherwise, choose the best action based on policy

P = TypeVar('PolicyArgs', bound='PRLArgs')
class PRL(RL[P]):
    def __init__(self, env, args:P ,**kwargs):
        super().__init__(env=env, args=args, **kwargs)
    
    @torch.no_grad()
    def gae(self, td_delta, is_terninated):
        td_delta = td_delta.detach().numpy().flatten()
        is_terninated = is_terninated.detach().numpy().flatten()
        advantages_list = []
        advantage = 0.0
        for delta, done in zip(td_delta[::-1], is_terninated[::-1]):
            advantage = self.cfg.gamma * self.cfg.gae_lambda * advantage * (1 - done) + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.tensor(np.array(advantages_list), dtype=torch.float, device=self.device).unsqueeze(-1)