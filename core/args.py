from dataclasses import dataclass, field, fields
import numpy as np
import argparse
from core import *
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(name="ArgParser") 

parser = argparse.ArgumentParser(description='RL algorithm parameters')
## Required arguments ##
parser.add_argument('--env_name', type=str, help='Environment name')
parser.add_argument('--alg_name', type=str, help=f"Algorithm name, support {allModels}, DQN_Series support any combination of {DQN_Series}+DQN")
parser.add_argument('--config', type=str, required=False, help='Path to the recipe file')
## Common arguments for all classes ##
parser.add_argument('--mode', type=str, choices=["train", "test"], help="If test, will automatically use the parameter in results/ to run.")
parser.add_argument('--max_epochs', type=float, help='Maximum number of epochs')
parser.add_argument('--max_timesteps', type=float, help='Maximum number of timesteps')
parser.add_argument('--reward_threshold', type=float, help='Reward threshold for early stopping')
parser.add_argument('--early_stop', action='store_true', help='Enable early stopping when average_reward and optimal_reward both exceed threshold')
parser.add_argument('--baseline', type=float, help='Baseline value, east method for getting advantage')
parser.add_argument('--gamma', type=float, help='Discount factor')
parser.add_argument('--lr', type=float, help='Optimizer learning rate')
parser.add_argument('--h_size', type=int, help='Hidden layer size')
parser.add_argument('--window_size', type=int, help='Window size for running average')
parser.add_argument('--eval_freq', type=int, help='Every N timesteps, evaluate the model and then record')
parser.add_argument('--episode_eval_freq', type=int, help='If set, the evaluation will be done every N episodes, for providing view of episode learning curve')
# parser.add_argument('--report_freq', type=int, help='Reporting frequency of the evluation result, only take effect when greater than eval_freq')
parser.add_argument('--max_episode_steps', type=int, help='Maximum episode steps')
parser.add_argument('--eval_epochs', type=int, help='Number of evaluation epochs')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--norm_obs', action="store_true", help='Normalize observation')
parser.add_argument('--norm_reward', action="store_true", help='Normalize reward')

## VRL-specific arguments ##
parser.add_argument('--epsilon_start', type=float, help='Starting epsilon value')
parser.add_argument('--epsilon_end', type=float, help='Final epsilon value')
parser.add_argument('--epsilon_decay', type=float, help='Epsilon decay rate')
parser.add_argument('--epsilon_decay_flag', action='store_true', help='Enable epsilon decay')
parser.add_argument('--sync_freq', type=int, help='Synchronization steps')
parser.add_argument('--memory_size', type=int, help='Replay memory size')
## DQN-specific arguments ##
parser.add_argument('--std_init', type=float, help='Initial standard deviation for NoisyDQN')
## PRL-specific arguments ##
parser.add_argument('--gae_lambda', type=float, help='Lambda parameter for GAE, 1 is equivalent to classic advantage')
parser.add_argument('--norm_advantage', action="store_true", help='Normalize advantages')
## A2C-specific arguments ##
parser.add_argument('--actor_lr', type=float, help='Actor learning rate')
parser.add_argument('--critic_lr', type=float, help='Critic learning rate')
parser.add_argument('--entropy_coef', type=float, help='Entropy coefficient')
parser.add_argument('--horizon', type=int, help='Update every N steps')
## PPO-specific arguments ##
parser.add_argument('--update_times', type=int, help='Update times in one updating')
parser.add_argument('--eps_clip', type=float, help='Clip for PPO')
parser.add_argument('--entropy_decay', type=float, help='Decay rate of entropy_coef')
parser.add_argument('--target_kl', type=float, help='Target KL divergence for early stopping')   
## DDPG-specific arguments ##
parser.add_argument('--noise_type', type=str, choices=["Gaussian", "OU"], help='Noise type for exploration')


@dataclass(kw_only=True, frozen=False)
class BasicArgs:
    env_name:str = ""
    alg_name:str = ""
    mode:str = "train"
    max_epochs:float = np.inf
    max_timesteps:float = np.inf
    reward_threshold:float = None
    early_stop:bool = True
    baseline:float = 0
    gamma:float = 0.99
    h_size:int = 64
    window_size:int = 10
    eval_freq:int = 100
    episode_eval_freq:int = None
    # report_freq:int = 100
    max_episode_steps:int = None
    eval_epochs:int = 10
    norm_obs:bool = False
    norm_reward:bool = False

    def __post_init__(self):
        base_args, _ = parser.parse_known_args()
        if base_args.config:
            recipe_path = base_args.config
            if not os.path.exists(recipe_path):
                raise FileNotFoundError(f"Recipe file {recipe_path} not found.")
            else:
                para_dict = self._read_recipe(recipe_path)
                for k, v in para_dict.items():
                    if k in self.__dict__:
                        setattr(self, k, v)
                    else:
                        logger.warning(f"{k} is not a valid argument. Ignoring it.")
                else:
                    logger.info(f"Loading recipe file {recipe_path} successfully, unspecified parameters will use default values.")
        else:
            args = parser.parse_args()
            for k, v in vars(args).items():
                if k in self.__dict__:
                    if v is not None:
                        setattr(self, k, v)
            else:
                logger.info("Loading command line arguments successfully, unspecified parameters will use default values.")
            
    def _read_recipe(self, recipe_path):
        with open(recipe_path, "r") as f:
            para_dict = yaml.safe_load(open(recipe_path, "r"))
        return para_dict

@dataclass(kw_only=True, frozen=False)
class VRLArgs(BasicArgs):
    epsilon_start:float = 1.0
    epsilon_end:float = 0.01
    epsilon_decay:float = 0.002
    epsilon_decay_flag:bool = True
    lr:float = 1e-4

@dataclass(kw_only=True, frozen=False)
class PRLArgs(BasicArgs):
    norm_advantage:bool = False

@dataclass(kw_only=True, frozen=False)
class REINFORCEArgs(BasicArgs):
    lr:float = 1e-4

@dataclass(kw_only=True, frozen=False)
class A2CArgs(PRLArgs):
    actor_lr:float = 3e-4
    critic_lr:float = 2e-4
    entropy_coef:float = 1e-3
    horizon:int = 128
    gae_lambda:float = 0.95
    batch_size:int = 64

@dataclass(kw_only=True, frozen=False)
class PPOArgs(A2CArgs):
    update_times:int = 10
    eps_clip:float = 0.2
    entropy_decay:float = 0.99
    target_kl:float = None

@dataclass(kw_only=True, frozen=False)
class DQNArgs(VRLArgs):
    sync_freq:int = 64   # sync target network with policy network
    memory_size:int = 6000
    batch_size:int = 32
    # NoisyDQN
    std_init:float = 0.5

@dataclass(kw_only=True, frozen=False)
class DDPGArgs(BasicArgs):
    actor_lr:float = 3e-4
    critic_lr:float = 2e-4
    memory_size:int = 10000
    batch_size:int = 256
    noise_type:str = "Gaussian"
    tau:float = 0.01 # soft update

ARGS_MAP = {
    "Q_Learning": VRLArgs,
    "Sarsa": VRLArgs,
    "REINFORCE": REINFORCEArgs,
    "A2C": A2CArgs,
    "PPO": PPOArgs,
    "DQN": DQNArgs,
    "DDPG": DDPGArgs,
}