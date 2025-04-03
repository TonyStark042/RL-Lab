from dataclasses import dataclass, field, fields
import numpy as np
import argparse
from core import *
import json
import yaml

parser = argparse.ArgumentParser(description='RL algorithm parameters')

# Required arguments
parser.add_argument('--env_name', type=str, default='', help='Environment name')
parser.add_argument('--alg_name', type=str, default='', help=f"Algorithm name, support {allModels}, DQN_Series support any combination of {DQN_Series}+DQN")

# Common arguments for all classes
parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="If test, will automatically use the parameter in results/ to run.")
parser.add_argument('--train_mode', type=str, default="timestep", choices=["episode", "timestep"])
parser.add_argument('--max_epochs', type=float, default=np.inf, help='Maximum number of epochs')
parser.add_argument('--max_timesteps', type=float, default=np.inf, help='Maximum number of timesteps')
parser.add_argument('--reward_threshold', type=float, default=None, help='Reward threshold for early stopping')
parser.add_argument('--early_stop', action='store_true', help='Enable early stopping when average_reward and optimal_reward both exceed threshold')
parser.add_argument('--baseline', type=float, default=0, help='Baseline value, east method for getting advantage')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--lr', type=float, default=1e-4, help='Optimizer learning rate')
parser.add_argument('--h_size', type=int, default=32, help='Hidden layer size')
parser.add_argument('--window_size', type=int, default=10, help='Window size for running average')
parser.add_argument('--timestep_freq', type=int, default=100, help='Every N timesteps, evaluate the model and then record')
parser.add_argument('--report_freq', type=int, default=100, help='Reporting frequency')
parser.add_argument('--max_episode_steps', type=int, default=None, help='Maximum episode steps')
parser.add_argument('--eval_epochs', type=int, default=10, help='Maximum episode steps')

# VRL-specific arguments
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon value')
parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon value')
parser.add_argument('--epsilon_decay', type=float, default=0.002, help='Epsilon decay rate')
parser.add_argument('--epsilon_decay_flag', action='store_true', help='Enable epsilon decay')
parser.add_argument('--sync_freq', type=int, default=64, help='Synchronization steps')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--memory_size', type=int, default=6000, help='Replay memory size')
    # DQN-specific arguments
parser.add_argument('--std_init', type=float, default=0.5, help='Initial standard deviation for NoisyDQN')

# PRL-specific arguments
parser.add_argument('--is_gae', action='store_true', help='Use Generalized Advantage Estimation')
parser.add_argument('--lmbda', type=float, default=0.95, help='Lambda parameter for GAE')
    # A2C-specific arguments
parser.add_argument('--actor_lr', type=float, default=1e-3, help='Actor learning rate')
parser.add_argument('--critic_lr', type=float, default=1e-4, help='Critic learning rate')
parser.add_argument('--entropy_coef', type=float, default=0.001, help='Entropy coefficient')
        # PPO-specific arguments
parser.add_argument('--update_freq', type=int, default=200, help='Update every N steps')
parser.add_argument('--update_times', type=int, default=10, help='Update times in one updating')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip for PPO')

args = parser.parse_args()


@dataclass(kw_only=True, frozen=False)
class Args:
    env_name:str = args.env_name
    mode:str = args.mode
    train_mode:str = args.train_mode
    max_epochs:float = args.max_epochs
    max_timesteps:float = args.max_timesteps
    reward_threshold:float = args.reward_threshold
    early_stop:bool = args.early_stop
    baseline:float = args.baseline
    gamma:float = args.gamma
    lr:float = args.lr
    h_size:int = args.h_size
    window_size:int = args.window_size
    timestep_freq:int = args.timestep_freq
    report_freq:int = args.report_freq
    alg_name:str = args.alg_name
    max_episode_steps:int = args.max_episode_steps
    eval_epochs:int = args.eval_epochs

    def __post_init__(self):
        pass

@dataclass(kw_only=True, frozen=False)
class VRLArgs(Args):
    epsilon_start:float = args.epsilon_start
    epsilon_end:float = args.epsilon_end
    epsilon_decay:float = args.epsilon_decay
    epsilon_decay_flag:bool = args.epsilon_decay_flag

@dataclass(kw_only=True, frozen=False)
class PRLArgs(Args):
    is_gae:bool = args.is_gae
    lmbda:float = args.lmbda

@dataclass(kw_only=True, frozen=False)
class REINFORCEArgs(Args):
    pass

@dataclass(kw_only=True, frozen=False)
class A2CArgs(PRLArgs):
    actor_lr:float = args.actor_lr
    critic_lr:float = args.critic_lr
    entropy_coef:float = args.entropy_coef

@dataclass(kw_only=True, frozen=False)
class PPOArgs(A2CArgs):
    update_freq:int = args.update_freq # update every 200 steps
    update_times:int = args.update_times
    eps_clip:float = args.eps_clip

@dataclass(kw_only=True, frozen=False)
class DQNArgs(VRLArgs):
    sync_freq:int = args.sync_freq   # sync target network with policy network
    batch_size:int = args.batch_size
    memory_size:int = args.memory_size 
    pass

@dataclass(kw_only=True, frozen=False)
class NoisyDQNArgs(DQNArgs):
    std_init:float = args.std_init


if args.mode == "train":
    if "DQN" in args.alg_name:
        if "Noisy" in args.alg_name:
            args = NoisyDQNArgs()
        else:
            args = DQNArgs()
    elif args.alg_name in ["Q-Learning", "Sarsa"]:
        args = VRLArgs()
    elif "REINFORCE" in args.alg_name:
        args = REINFORCEArgs()
    elif "A2C" in args.alg_name:
        args = A2CArgs()
    elif "PPO" in args.alg_name:
        args = PPOArgs()
    else:
        args = Args()
else:
    training_hyperparameters = yaml.safe_load(open(f"results/{args.alg_name}/{args.env_name}_{args.train_mode}/recipe.yaml", "r"))
    if "DQN" in args.alg_name:
        if "Noisy" in args.alg_name:
            args = NoisyDQNArgs(**training_hyperparameters)
        else:
            args = DQNArgs(**training_hyperparameters)
    elif args.alg_name in ["Q-Learning", "Sarsa"]:
        args = VRLArgs(**training_hyperparameters)
    elif "REINFORCE" in args.alg_name:
        args = REINFORCEArgs(**training_hyperparameters)
    elif "A2C" in args.alg_name:
        args = A2CArgs(**training_hyperparameters)
    elif "PPO" in args.alg_name:
        args = PPOArgs(**training_hyperparameters)
    else:
        args = Args(**training_hyperparameters)

