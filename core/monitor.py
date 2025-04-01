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
        evalute the model reward situation around rencent <window_size> episode, differs from timestep_report, epoch_report's average_n_reward uses training data instead of evalating data
        Args:
            epoch: The current episode;
        """
        if len(self.agent.epoch_record) <= self.agent.window_size:
            avg_n_reward = np.mean(sum(self.agent.epoch_record) / len(self.agent.epoch_record))
        else:
            avg_n_reward = np.mean(sum(self.agent.epoch_record[-self.agent.window_size:]) / self.agent.window_size)

        evaluate_reward = self.agent.evaluate()
        self.agent.timestep_record["rewards"].append(evaluate_reward)
        self.agent.timestep_record["timesteps"].append(self.agent.epoch)
        mean_evaluate_reward = np.mean(evaluate_reward)
        
        if mean_evaluate_reward >= self.agent.optimal_reward and self.agent.alg_name not in noDeepLearning:
            self.agent.optimal_reward = mean_evaluate_reward
            state_dict = getattr(self.agent, self.agent.model_name)
            self.agent.best = state_dict

        if self.agent.epoch % self.agent.report_freq == 0:
            message = f"Episode: {self.agent.epoch} | Average_{self.agent.window_size}_reward: {avg_n_reward:.3f} | Evaluation reward: {mean_evaluate_reward} | History optimal: {self.agent.optimal_reward} "
            for k,v in kwargs.items():
                message += f"| {k}: {v:.3f} "
            self.agent.logger.info(message)
    
        if self.agent.early_stop:
            if avg_n_reward >= self.agent.reward_threshold and mean_evaluate_reward >= self.agent.reward_threshold:
                self.agent.logger.info(f"Converged at epoch: {self.agent.epoch}, final optimal reward: {mean_evaluate_reward}")
                return True
                
        return False
    
    def timestep_report(self, **kwargs) -> bool:
        """
        evalute the model reward situation every self.agent.timestep_freq in rencent <window_size> timestep, and save the best model, checking early stop
        """

        if self.agent.train_mode == "timestep" and self.agent.timestep % self.agent.timestep_freq == 0:
            evaluate_reward = self.agent.evaluate()
            mean_evaluate_reward = np.mean(evaluate_reward)
            self.agent.timestep_record["rewards"].append(evaluate_reward)
            self.agent.timestep_record["timesteps"].append(self.agent.timestep)
            if len(self.agent.timestep_record["rewards"]) <= self.agent.window_size:
                avg_n_reward = np.mean(sum(self.agent.timestep_record["rewards"]) / len(self.agent.timestep_record["rewards"]))
            else:
                avg_n_reward = np.mean(sum(self.agent.timestep_record["rewards"][-self.agent.window_size:]) / self.agent.window_size)
        
            if mean_evaluate_reward >= self.agent.optimal_reward and self.agent.alg_name not in noDeepLearning:
                self.agent.optimal_reward = mean_evaluate_reward
                state_dict = getattr(self.agent, self.agent.model_name)
                self.agent.best = state_dict

            if self.agent.timestep % self.agent.report_freq == 0:
                message = f"Timestep: {self.agent.timestep} | Average_{self.agent.window_size}_reward: {avg_n_reward:.3f} | Evaluation reward: {mean_evaluate_reward} | History optimal: {self.agent.optimal_reward} "
                for k,v in kwargs.items():
                    message += f"| {k}: {v:.3f}"
                self.agent.logger.info(message)
        
            if self.agent.early_stop:
                if avg_n_reward >= self.agent.reward_threshold and mean_evaluate_reward >= self.agent.reward_threshold:
                    self.agent.logger.info(f"Converged at timestep: {self.agent.timestep}, final optimal reward: {mean_evaluate_reward}")
                    return True
            return False
        else:
            return False
        
    def learning_curve(self, mode=Literal["episode", "timestep"]):
        """
        Plot and save the learning curve with optional moving average and visualization, showing the evaluation reward, instead of training reward, the std looks low beacause uses moving average std.
        """
        result_path = self._check_dir()
        plt.figure(figsize=(10, 6))
        if hasattr(self.agent, 'reward_threshold') and self.agent.reward_threshold not in (None, np.inf):
            plt.axhline(y=self.agent.reward_threshold, color='r', linestyle='--', label='Reward Threshold')
        plt.axhline(y=self.agent.optimal_reward, color='green', linestyle='--', label='Optimal Reward')

        name = self.agent.train_mode
        if mode == "episode":
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
        elif mode == "timestep":
            plt.xlabel('timesteps')
            plt.ylabel('Rewards')
        X = self.agent.timestep_record['timesteps']
        Y_arrays = np.array(self.agent.timestep_record['rewards'])
        Y_mean = np.array([np.mean(y) for y in Y_arrays])
        # Y_std = np.array([np.std(y) for y in Y_arrays])
        # plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, color='gray', alpha=0.2)

        plt.plot(X, Y_mean, alpha=0.3, color='gray', label='Raw Rewards')
        moving_avg = []
        moving_std = []
        for index, reward_array in enumerate(Y_arrays):
            if index < self.agent.window_size:
                moving_avg.append(reward_array.sum())
                moving_std.append(reward_array.std())
            else:
                sliding_window = Y_arrays[index - self.agent.window_size + 1:index + 1]
                sliding_window = sliding_window.mean(axis=1)
                sliding_avg = sliding_window.mean()
                sliding_std = sliding_window.std()
                moving_avg.append(sliding_avg)
                moving_std.append(sliding_std)
        moving_avg = np.array(moving_avg)
        moving_std = np.array(moving_std)

        plt.plot(X, moving_avg, color='blue', label=f'Moving Average (n={self.agent.window_size})')
        plt.fill_between(X, moving_avg - moving_std, moving_avg + moving_std, color='blue', alpha=0.2)
        
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
            if self.agent.early_stop and self.agent.reward_threshold != np.inf:
                self.agent.logger.warning("max_epochs and max_timesteps is not specified, the training will continue until reward_threshold are met.")
            else:
                raise ValueError("Please specify the number of max_epochs or max_timesteps for training, or set the early_stop to True and set the reward_threshold.") 

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