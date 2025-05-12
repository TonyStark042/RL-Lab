from matplotlib import pyplot as plt
import numpy as np
import os
from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from core.baseModule import RL
noDeepLearning = ["Q_Learning", "Sarsa"]

class RLMonitor:
    def __init__(self, RLinstace: "RL"):
        self.agent = RLinstace
        self.cfg = RLinstace.cfg
    
    @classmethod
    def check_args(cls, RLinstace: "RL"):
        """
        Check some key argument values and print all hyperparameters.
        """
        cfg = RLinstace.cfg
        if cfg.reward_threshold < 0:
            raise ValueError("reward_threshold must be a non-negative number.")
        if cfg.max_episode_steps is None:
            RLinstace.logger.warning("max_episode_steps is not specified, please make sure the env must have an end.")
        if RLinstace.model_names is None and cfg.alg_name not in noDeepLearning:
            raise ValueError("Please specify the name of learnable policy net or Q net, which will be used in saving and updating best model")
        if cfg.alg_name is None:
            raise ValueError("Please specify the name of the algorithm, which will be used in saving model and learning curve")
        if cfg.max_epochs is np.inf and cfg.max_timesteps is np.inf:
            if cfg.early_stop and cfg.reward_threshold != np.inf:
                RLinstace.logger.warning("max_epochs and max_timesteps is not specified, the training will continue until reward_threshold are met.")
            else:
                raise ValueError("Please specify the number of max_epochs or max_timesteps for training, or set the early_stop to True and set the reward_threshold.") 
        monitor = cls(RLinstace)
        monitor._print_config()
        monitor._check_dir()
        return monitor

    def _print_config(self):
        """Print agent configuration in a formatted table"""
        print(''.join(['=']*140))
        tplt = "{:^30}\t{:^60}\t{:^30}"
        print(tplt.format("Arg","Value","Type"))
        
        for k, v in vars(self.cfg).items():
            print(tplt.format(str(k), str(v).replace(" ", ''), str(type(v))))
        
        print(''.join(['=']*140))

    def timestep_report(self, report_dict={}) -> bool:
        """
        evalute the model reward situation every self.agent.eval_freq in rencent <window_size> timestep, and save the best model, checking early stop
        """
        total_timestep = self.agent.timestep.sum().item()
        if total_timestep % self.cfg.eval_freq == 0:
            evaluate_reward = self.agent.evaluate()
            mean_evaluate_reward = np.mean(evaluate_reward)
            self.agent.timestep_eval["rewards"].append(evaluate_reward)
            self.agent.timestep_eval["timesteps"].append(total_timestep)
            if len(self.agent.timestep_eval["rewards"]) <= self.cfg.window_size:
                avg_n_reward = np.mean(sum(self.agent.timestep_eval["rewards"]) / len(self.agent.timestep_eval["rewards"]))
            else:
                avg_n_reward = np.mean(sum(self.agent.timestep_eval["rewards"][-self.cfg.window_size:]) / self.cfg.window_size)
        
            if mean_evaluate_reward >= self.agent.optimal_reward and self.cfg.alg_name not in noDeepLearning:
                self.agent.optimal_reward = mean_evaluate_reward
                for net_name in self.agent.model_names:
                    model = getattr(self.agent, net_name)
                    self.agent.best[net_name] = model

            # if self.agent.timestep % self.agent.report_freq == 0:
            message = f"Timestep: {total_timestep} | Avg_{self.cfg.window_size}_R: {avg_n_reward:.3f} | Eval R: {mean_evaluate_reward: .3f} | Best R: {self.agent.optimal_reward: .3f} | T: {self._seconds_to_hms(self.agent.training_time)}"
            for k,v in report_dict.items():
                message += f"| {k}: {v:.3f}"
            self.agent.logger.info(message)
        
            if self.cfg.early_stop:
                if avg_n_reward >= self.cfg.reward_threshold and mean_evaluate_reward >= self.cfg.reward_threshold:
                    self.agent.logger.info(f"Converged at timestep: {total_timestep}, final optimal reward: {mean_evaluate_reward: .3f}")
                    return True
            return False
        else:
            return False
    
    def episode_evaluate(self):
        totoal_episode = self.agent.episode.sum().item()
        if self.cfg.episode_eval_freq is not None and (totoal_episode / self.cfg.num_envs) % self.cfg.episode_eval_freq == 0:
            evaluate_reward = self.agent.evaluate()
            self.agent.episode_eval["rewards"].append(evaluate_reward)
            self.agent.episode_eval["episodes"].append(totoal_episode)

    def learning_curve(self, mode=Literal["episode", "timestep"]):
        """
        Plot and save the learning curve with optional moving average and visualization, showing the evaluation reward, instead of training reward, the std looks low beacause uses moving average std.
        """
        save_dir = self._check_dir()
        plt.figure(figsize=(10, 6))
        plt.axhline(y=self.agent.optimal_reward, color='green', linestyle='--', label='Optimal Reward')

        name = mode
        if mode == "episode":
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
            record = self.agent.episode_eval
            X = record['episodes']
        elif mode == "timestep":
            plt.xlabel('timesteps')
            plt.ylabel('Rewards')
            record = self.agent.timestep_eval
            X = record['timesteps']
        Y_arrays = np.array(record['rewards'])
        Y_mean = np.array([np.mean(y) for y in Y_arrays])
        # Y_std = np.array([np.std(y) for y in Y_arrays])
        # plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, color='gray', alpha=0.2)

        plt.plot(X, Y_mean, alpha=0.3, color='gray', label='Raw Rewards')
        moving_avg = []
        moving_std = []
        for index, reward_array in enumerate(Y_arrays):
            if index < self.cfg.window_size:
                sliding_window = Y_arrays[:index + 1]
            else:
                sliding_window = Y_arrays[index - self.cfg.window_size + 1:index + 1]
                # sliding_window = sliding_window.mean(axis=1)
            sliding_avg = sliding_window.mean()
            sliding_std = sliding_window.std()
            moving_avg.append(sliding_avg)
            moving_std.append(sliding_std)
        moving_avg = np.array(moving_avg)
        moving_std = np.array(moving_std)

        plt.plot(X, moving_avg, color='blue', label=f'Moving Average (n={self.cfg.window_size})')
        plt.fill_between(X, moving_avg - moving_std, moving_avg + moving_std, color='blue', alpha=0.2)
        
        save_path = os.path.join(save_dir, f"{name}.png")
        plt.grid(True, alpha=0.3)
        plt.title(self.agent.cfg.env_name)
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        self.agent.logger.info(f"Learning curve has been saved to {save_path}")   
    
    def _check_dir(self):
        """
        Check if there is the model's saving directory.
        """
        if self.cfg.save_dir is None:
            self.cfg.save_dir = os.path.join("results", self.cfg.alg_name, f"{self.cfg.env_name}")
        if not os.path.exists(self.cfg.save_dir):
            os.makedirs(self.cfg.save_dir, exist_ok=True)
        return self.cfg.save_dir

    def _seconds_to_hms(self, seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
