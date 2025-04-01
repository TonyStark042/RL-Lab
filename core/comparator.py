import gymnasium as gym
from models import *
from core.args import *
import logging
import torch.multiprocessing as mp
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import tempfile
import pickle
import shutil
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def _train_wrapper(agent_file):
    """
    Wrapper function to train an agent in a separate process.
    
    Args:
        agent_file: Path to the pickled agent file
    """
    with open(agent_file, 'rb') as f:
        agent = pickle.load(f)
    agent.train()
    with open(agent_file, 'wb') as f:
        agent = pickle.dump(agent, f)


class Comparator:
    """
    A class for comparing different RL algorithms on the same environment.
    """
    def __init__(self, **kwargs):
        """
        Initialize the comparator.
        """
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.agents = {}
        self.logger = logging.getLogger("Comparator")
    
    def add_algorithm(self, *tuples):
        """
        Add an algorithm to compare.
        """
        for name, agent in tuples:
            self.agents[name] = agent
    
    def train_all(self, parallel=True, n_processes=None):
        """
        Train all added algorithms, optionally in parallel.
        
        Args:
            parallel (bool): Whether to train algorithms in parallel
            n_processes (int, optional): Number of parallel processes to use.
                                        If None, uses all available CPU cores.
        """
        self.results_dir = os.path.join("results", "comparisons", "_".join(self.agents.keys()), self.env_name)
        if not parallel:
            for name, agent in self.agents.items():
                self.logger.info(f"Start training {name}...")
                
                start_time = datetime.now()
                agent.train()
                end_time = datetime.now()
                
                self.training_times[name] = (end_time - start_time)
                
                self.logger.info(f"Finished training {name} in {self.training_times[name]:.2f} seconds, final reward: {agent.optimal_reward:.2f}")
                print("-" * 140)
            return
        else:
            temp_dir = tempfile.mkdtemp()

            if n_processes is None:
                n_processes = min(mp.cpu_count(), len(self.agents))
            self.logger.info(f"Starting parallel training with {n_processes} processes")
            processes = []
            
            for name, agent in self.agents.items():
                self.logger.info(f"Preparing {name} for parallel training...")
                agent_file = os.path.join(temp_dir, f"{name}_agent.pkl")
                with open(agent_file, 'wb') as f:
                    pickle.dump(agent, f)
                p = mp.Process(target=_train_wrapper, args=(agent_file,))
                processes.append((name, p))
                p.start()
                self.logger.info(f"Started training process for {name}, process ID: {p.pid}")
            
            for name, p in processes:
                p.join()
                with open(os.path.join(temp_dir, f"{name}_agent.pkl"), 'rb') as f:
                    trained_agent = pickle.load(f)
                    self.agents[name] = trained_agent
                self.logger.info(f"Training process for {name} completed, time consumed {self.agents[name].training_time:.2f}s")
                print("-" * 140)
            
            self.logger.info("Parallel training completed for all algorithms")
    
    def evaluate_all(self, num_episodes=10):
        """
        Evaluate all trained algorithms.
        
        Args:
            num_episodes (int): Number of episodes to evaluate each algorithm
            
        Returns:
            dict: Dictionary mapping algorithm names to average rewards
        """
        results = {}
        
        for name, agent in self.agents.items():
            agent.eval_episodes = num_episodes
            avg_reward = agent.evaluate()
            results[name] = avg_reward
            
        return results
    
    def learning_curve(self, save=True):
        """
        Plot learning curves for all algorithms.
        
        Args:
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors

        if self.train_mode == "episode":
            plt.xlabel('Episodes')
        elif self.train_mode == "timestep":
            plt.xlabel('timesteps')

        for index, (name, agent) in enumerate(self.agents.items()):
            X = agent.timestep_record['timesteps']
            Y_arrays = np.array(agent.timestep_record['rewards'])
            color = colors[index]

            moving_avg = []
            moving_std = []
            for index, reward_array in enumerate(Y_arrays):
                if index < self.window_size:
                    moving_avg.append(reward_array.sum())
                    moving_std.append(reward_array.std())
                else:
                    sliding_window = Y_arrays[index - self.window_size + 1:index + 1]
                    sliding_window = sliding_window.mean(axis=1)
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
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
        if save:
            plt.savefig(os.path.join(self.results_dir, f"{agent.train_mode}.png") , bbox_inches='tight', dpi=300)
            self.logger.info(f"Comparison Learning curve has been saved to {self.results_dir}")
    
    def save_all(self):
        """
        Save all trained algorithms.
        """
        with open(os.path.join(self.results_dir, 'comparator.pkl'), 'wb') as f:
            pickle.dump(self, f)
        shutil.copyfile("compare.yaml", os.path.join(self.results_dir, 'compare.yaml'))
        self.logger.info(f"Compatator object and arguments has been saved to {self.results_dir}")