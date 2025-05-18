import logging
import multiprocessing as mp
import torch.multiprocessing as mp
import os
import numpy as np
from matplotlib import pyplot as plt
import shutil
from core.baseModule import RL
import yaml
from omegaconf import OmegaConf
import gymnasium as gym
from core.args import BasicArgs
from utils import create_agent
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def _train_wrapper(args:dict, queue:mp.Queue=None):
    """
    Wrapper function to train an agent in a separate process.
    """ 
    agent = create_agent(args.get("alg_name"), args, multi_env=True, load=False)
    agent.train()
    agent.logger.info(f"Training process for {agent.cfg.alg_name} completed, time consumed {agent.training_time:.2f}s")
    agent.save()
    result = {agent.cfg.alg_name: agent.timestep_eval}
    queue.put(result)

class Comparator:
    """
    A class for comparing different RL algorithms on the same environment.
    """
    def __init__(self, env_name:str=None, save_dir:str=None, recipe_path:str=None):
        """
        Initialize the comparator.
        """
        self.logger = logging.getLogger("Comparator")
        self.agents: dict[str, RL]
        self.agents_arg: dict[str, BasicArgs] = dict()
        self.save_dir: str = save_dir
        self.env_name: str = env_name
        self.recipe_path: str = recipe_path
        self.window_size: int = None
    
    @classmethod
    def initialize(cls, recipe_path:str="recipes/compare.yaml", save_dir:str=None):
        """
        Add an algorithm to compare.
        """    
        params = yaml.safe_load(open(recipe_path, "r"))
        common_args = OmegaConf.create(params.pop("common_args"))
        env_name = common_args.get("env_name")
        com = cls(env_name=env_name, save_dir=save_dir, recipe_path=recipe_path)

        for alg_name, args in params.items():
            args["alg_name"] = alg_name
            model_args = OmegaConf.merge(common_args, args)
            model_args = OmegaConf.to_object(model_args)
            com.agents_arg[alg_name] = model_args
        
        com.save_dir = save_dir if save_dir is not None else os.path.join("results", "comparisons", "_".join(com.agents_arg.keys()), env_name)
        com.window_size = model_args.get("window_size", 10)
        com.save_config()
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
            n_processes = min(mp.cpu_count(), len(self.agents_arg))
        self.logger.info(f"Starting parallel training with {n_processes} processes")

        queue = mp.Queue()
        processes = list()
        for alg_name, args in self.agents_arg.items():
            args["save_dir"] = os.path.join(self.save_dir, alg_name)
            process = mp.Process(target=_train_wrapper, args=(args,queue), daemon=False)
            processes.append(process)
            process.start()

        results = list()
        for process in processes:
            process.join()
            results.append(queue.get())

        self.traning_result = {k: v for d in results for k, v in d.items()}
        self.logger.info("Parallel training completed for all algorithms")

    def learning_curve(self, save=True):
        """
        Plot learning curves for all algorithms.
        
        Args:
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10.colors
        plt.xlabel('timesteps')

        for index, (name, result) in enumerate(self.traning_result.items()):
            X = result['timesteps']
            Y_arrays = np.array(result['rewards'])
            color = colors[index]

            moving_avg = []
            moving_std = []
            for index, reward_array in enumerate(Y_arrays):
                if index < self.window_size:
                    sliding_window = Y_arrays[:index + 1]
                else:
                    sliding_window = Y_arrays[index - self.window_size + 1:index + 1]
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
        
        if save:
            plt.savefig(os.path.join(self.save_dir, "comparison.png") , bbox_inches='tight', dpi=300)
            self.logger.info(f"Comparison Learning curve has been saved to {self.save_dir}")
    
    def save_config(self, params=None):
        """
        Save the configuration of all algorithms to a YAML file.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        shutil.copy(self.recipe_path, self.save_dir)
    
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
        
        
        