import logging
from multiprocessing.pool import ThreadPool
import torch.multiprocessing as mp
import os
import numpy as np
from matplotlib import pyplot as plt
import shutil
from core.baseModule import RL
import yaml
from omegaconf import OmegaConf
import gymnasium as gym
from utils import create_agent
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def _train_wrapper(agent:RL):
    """
    Wrapper function to train an agent in a separate process.
    """
    agent.train()
    agent.logger.info(f"Training process for {agent.cfg.alg_name} completed, time consumed {agent.training_time:.2f}s")
    return agent

class Comparator:
    """
    A class for comparing different RL algorithms on the same environment.
    """
    def __init__(self, env_name:str=None, save_dir:str=None):
        """
        Initialize the comparator.
        """
        self.logger = logging.getLogger("Comparator")
        self.agents: dict[str, RL]
        self.save_dir: str = save_dir
        self.env_name: str = env_name
        self.recipe_path: str = None
    
    @classmethod
    def initialize(cls, recipe_path:str="recipes/compare.yaml", save_dir:str=None):
        """
        Add an algorithm to compare.
        """
        agents = dict()
    
        model_args = yaml.safe_load(open(recipe_path, "r"))
        common_args = model_args.pop("common_args")
        common_args = OmegaConf.create(common_args)
        for alg_name, args in model_args.items():
            agents[alg_name] = create_agent(alg_name, args, common_args)
        
        env_name = common_args.get("env_name")
        com = cls(env_name=env_name, save_dir=save_dir)
        com.agents = agents
        com.recipe_path = recipe_path
        com.save_dir = save_dir if save_dir is not None else os.path.join("results", "comparisons", "_".join(com.agents.keys()), env_name)
        return com

    @classmethod
    def load_algorithms(cls, save_dir:str=None):
        """Loading a batch of algorithms from a comaprison directory."""

        agents = dict()
        for item in os.listdir(save_dir):
            if os.path.isdir(item):
                yaml_path = os.path.join(item, "recipe.yaml")   
                model_args = yaml.safe_load(open(yaml_path, "r"))
                alg_name = model_args.pop("alg_name")
                agent = create_agent(alg_name, model_args, multi_env=True, load=True)
                agents[alg_name] = agent

        com = cls(env_name=model_args.get("env_name"), save_dir=save_dir)
        com.agents = agents
        return com
    
    def train_all(self, n_processes=None):
        """
        Train all added algorithms, optionally in parallel.
        
        Args:
            parallel (bool): Whether to train algorithms in parallel
            n_processes (int, optional): Number of parallel processes to use.
                                        If None, uses all available CPU cores.
        """
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(self.agents))
        self.logger.info(f"Starting parallel training with {n_processes} processes")
        with ThreadPool(n_processes) as pool:
            agents = pool.map(_train_wrapper, [args for args in self.agents.values()])
        self.agents = {i.cfg.alg_name:  i for i in agents}
        self.logger.info("Parallel training completed for all algorithms")
    
    def evaluate_all(self):
        """
        Evaluate all trained algorithms.
        
        Args:
            num_episodes (int): Number of episodes to evaluate each algorithm
            
        Returns:
            dict: Dictionary mapping algorithm names to average rewards
        """
        results = {}
        
        for name, agent in self.agents.items():
            avg_reward = agent.evaluate()
            self.logger.info(f"Evaluating {name} in {agent.cfg.env_name}, Average {agent.cfg.eval_epochs} reward {np.mean(avg_reward):.3f}, Standard deviation {np.std(avg_reward):.3f}")
            results[name] = avg_reward
        
        self.test_results = results
        return results
    
    def learning_curve(self, save=True):
        """
        Plot learning curves for all algorithms.
        
        Args:
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        plt.xlabel('timesteps')

        for index, (name, agent) in enumerate(self.agents.items()):
            X = agent.timestep_eval['timesteps']
            Y_arrays = np.array(agent.timestep_eval['rewards'])
            color = colors[index]

            moving_avg = []
            moving_std = []
            for index, reward_array in enumerate(Y_arrays):
                if index < agent.cfg.window_size:
                    sliding_window = Y_arrays[:index + 1]
                else:
                    sliding_window = Y_arrays[index - agent.cfg.window_size + 1:index + 1]
                # sliding_window = sliding_window.mean(axis=1)
                sliding_avg = sliding_window.mean()
                sliding_std = sliding_window.std()
                moving_avg.append(sliding_avg)
                moving_std.append(sliding_std)
            moving_avg = np.array(moving_avg)
            moving_std = np.array(moving_std)

            plt.plot(X, moving_avg, color=color, label=name)
            plt.fill_between(X, moving_avg - moving_std, moving_avg + moving_std, color=color, alpha=0.2)
        
        plt.ylabel('Rewards')
        plt.title(f'Comparison on {self.env_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        if save:
            plt.savefig(os.path.join(self.save_dir, "comparison.png") , bbox_inches='tight', dpi=300)
            self.logger.info(f"Comparison Learning curve has been saved to {self.save_dir}")
    
    def save_all(self):
        """
        Save all trained algorithms.
        """
        for name, agent in self.agents.items():
            save_dir = os.path.join(self.save_dir, name)
            agent.cfg.save_dir = save_dir
            agent.save()
        shutil.copyfile(self.recipe_path, os.path.join(self.save_dir, 'compare.yaml'))
        self.logger.info(f"Compatator object and arguments has been saved to {self.save_dir}")