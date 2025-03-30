from core import noDeepLearning
from matplotlib import pyplot as plt
import numpy as np
import os
from typing import Literal
from datetime import datetime

class RLMonitor:
    def __init__(self, RLinstace):
        self.agent = RLinstace
        
    def epoch_report(self, **kwargs) -> bool:
        """
        evalute the model reward situation around rencent <window_size> episode, and save the best model, checking early stop
        Args:
            epoch: The current episode;
        """
        if len(self.agent.epoch_record) <= self.agent.window_size:
            avg_n_reward = sum(self.agent.epoch_record) / len(self.agent.epoch_record)
        else:
            avg_n_reward = sum(self.agent.epoch_record[-self.agent.window_size:])/self.agent.window_size

        optimal_reward = self.agent.evaluate()

        if optimal_reward >= self.agent.optimal_reward and self.agent.alg_name not in noDeepLearning:
            self.agent.optimal_reward = optimal_reward
            state_dict = getattr(self.agent, self.agent.model_name)
            self.agent.best = state_dict

        if self.agent.epoch % self.agent.report_freq == 0:
            message = f"Episode: {self.agent.epoch} | Average_{self.agent.window_size}_reward: {avg_n_reward:.3f} | Optimal reward: {optimal_reward} | History optimal: {self.agent.optimal_reward} "
            for k,v in kwargs.items():
                message += f"| {k}: {v:.3f} "
            self.agent.logger.info(message)
    
        if self.agent.early_stop:
            if avg_n_reward >= self.agent.reward_threshold and optimal_reward >= self.agent.reward_threshold:
                self.agent.logger.info(f"Converged at epoch: {self.agent.epoch}, final optimal reward: {optimal_reward}")
                return True
        return False
    
    def timestep_report(self, **kwargs) -> bool:
        """
        evalute the model reward situation every self.agent.timestep_freq in rencent <window_size> timestep, and save the best model, checking early stop
        """

        if self.agent.timestep_freq != None and self.agent.timestep % self.agent.timestep_freq == 0:
            optimal_reward = self.agent.evaluate()
            self.agent.timestep_record["rewards"].append(optimal_reward)
            self.agent.timestep_record["timesteps"].append(self.agent.timestep)
            if len(self.agent.timestep_record["rewards"]) <= self.agent.window_size:
                avg_n_reward = sum(self.agent.timestep_record["rewards"]) / len(self.agent.timestep_record["rewards"])
            else:
                avg_n_reward = sum(self.agent.timestep_record["rewards"][-self.agent.window_size:])/self.agent.window_size
        
            if optimal_reward >= self.agent.optimal_reward and self.agent.alg_name not in noDeepLearning:
                self.agent.optimal_reward = optimal_reward
                state_dict = getattr(self.agent, self.agent.model_name)
                self.agent.best = state_dict

            if self.agent.timestep % self.agent.report_freq == 0:
                message = f"Timestep: {self.agent.timestep} | Average_{self.agent.window_size}_reward: {avg_n_reward:.3f} | Optimal reward: {optimal_reward} | History optimal: {self.agent.optimal_reward} "
                for k,v in kwargs.items():
                    message += f"| {k}: {v:.3f}"
                self.agent.logger.info(message)
        
            if self.agent.early_stop:
                if avg_n_reward >= self.agent.reward_threshold and optimal_reward >= self.agent.reward_threshold:
                    self.agent.logger.info(f"Converged at timestep: {self.agent.timestep}, final optimal reward: {optimal_reward}")
                    return True
            return False
        else:
            return False
        

    def learning_curve(self, mode=Literal["episode", "timestep"]):
        """
        Plot and save the learning curve with optional moving average and visualization.
        Args:
            window_size (int): Size of the moving average window. Default is 10.
        """
        result_path = self._check_dir()
        plt.figure(figsize=(10, 6))
        if hasattr(self.agent, 'reward_threshold') and self.agent.reward_threshold not in (None, np.inf):
            plt.axhline(y=self.agent.reward_threshold, color='r', linestyle='--', label='Reward Threshold')
            plt.legend()
        plt.axhline(y=self.agent.optimal_reward, color='green', linestyle='--', label='Optimal Reward')

        name = self.agent.train_mode
        if mode == "episode":
            X = range(len(self.agent.epoch_record))
            Y = self.agent.epoch_record
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
        elif mode == "timestep":
            X = self.agent.timestep_record['timesteps']
            Y = self.agent.timestep_record['rewards']
            plt.xlabel('timesteps')
            plt.ylabel('Rewards')

        plt.plot(X, Y, alpha=0.3, color='gray', label='Raw Rewards')
        if len(Y) >= self.agent.window_size:
            moving_avg = np.convolve(Y, 
                                    np.ones(self.agent.window_size)/self.agent.window_size, 
                                    mode='valid') # only start calculating when epoch >= 10
            plt.plot(X[self.agent.window_size-1:], moving_avg, 
                    color='blue', label=f'Moving Average (n={self.agent.window_size})')
        
        save_path = os.path.join(result_path, f"{name}.png")
        plt.grid(True, alpha=0.3)
        plt.title(self.agent.env_name)
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        self.agent.logger.info(f"Learning curve has been saved to {save_path}")   
    
    def _check_dir(self):
        """
        Check if there is the model's saving directory.
        """
        # now = datetime.now()
        # time_str = now.strftime("%y%m%d%H%M")
        # result_path = os.path.join("results", self.agent.alg_name, f"{self.agent.env_name}_{time_str}")
        result_path = os.path.join("results", self.agent.alg_name, f"{self.agent.env_name}_{self.agent.train_mode}")
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        return result_path

    def _check_args(self):
        """
        Check some key argument values and print all hyperparameters.
        """
        if self.agent.reward_threshold is None or self.agent.reward_threshold == 0:
            self.agent.reward_threshold = np.inf if self.agent.env.spec.reward_threshold is None else self.agent.env.spec.reward_threshold
            self._print_config()
                    
        elif self.agent.reward_threshold < 0:
            raise ValueError("reward_threshold must be a non-negative number.")
        if self.agent.max_episode_steps is None:
            self.agent.logger.warning("max_episode_steps is not specified, please make sure the env must have an end.")
        if self.agent.model_name is None and self.agent.alg_name not in noDeepLearning:
            raise ValueError("Please specify the name of learnable policy net or Q net, which will be used in saving and updating best model")
        if self.agent.alg_name is None:
            raise ValueError("Please specify the name of the algorithm, which will be used in saving model and learning curve")
        if self.agent.max_epochs is np.inf and self.agent.max_timesteps is np.inf:
            if self.agent.early_stop:
                self.agent.logger.warning("max_epochs and max_timesteps is not specified, the training will continue until reward_threshold are met.")
            else:
                raise ValueError("Please specify the number of max_epochs for training, or set the early_stop to True.")
        

    def _print_config(self):
        """Print agent configuration in a formatted table"""
        print(''.join(['=']*140))
        tplt = "{:^30}\t{:^60}\t{:^30}"
        print(tplt.format("Arg","Value","Type"))
        
        for k, v in self.agent.__dict__.items():
            if k in ["env", "logger", "monitor", "custom_args"]:
                continue
            print(tplt.format(str(k), str(v).replace(" ", ''), str(type(v))))
        
        print(''.join(['=']*140))

    def save_results(self):
        pass
        # 实现结果保存