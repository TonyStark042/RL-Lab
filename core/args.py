from dataclasses import dataclass
from omegaconf import OmegaConf
import sys
import logging
from typing import Optional
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(name="ArgParser")

@dataclass(kw_only=True, frozen=False)
class BasicArgs():
    env_name:str = ""
    alg_name:str = ""
    max_epochs:float = np.inf
    max_timesteps:float = np.inf
    reward_threshold:Optional[float] = None
    early_stop:bool = True
    gamma:float = 0.99
    h_size:int = 64
    window_size:int = 10
    eval_freq:int = 100
    episode_eval_freq:Optional[int] = None
    max_episode_steps:Optional[int] = None
    eval_epochs:int = 10
    norm_obs:bool = False
    norm_reward:bool = False
    num_envs:int = 1
    save_dir:Optional[str] = None

@dataclass(kw_only=True, frozen=False)
class VRLArgs(BasicArgs):
    epsilon_start: float = 1.0
    epsilon_end:float = 0.01
    epsilon_decay:float = 0.002
    epsilon_decay_flag:bool = True
    lr:float = 1e-4
    expl_steps:int = 0

@dataclass(kw_only=True, frozen=False)
class PRLArgs(BasicArgs):
    norm_advantage:bool = False

@dataclass(kw_only=True, frozen=False)
class REINFORCEArgs(BasicArgs):
    lr:float = 1e-4
    baseline:float = 0

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
    target_kl:float = 0.05

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
    expl_steps:int = 0

class TD3Args(DDPGArgs):
    pass


ARGS_MAP = {
    "Q_Learning": VRLArgs,
    "Sarsa": VRLArgs,
    "REINFORCE": REINFORCEArgs,
    "A2C": A2CArgs,
    "PPO": PPOArgs,
    "DQN": DQNArgs,
    "DDPG": DDPGArgs,
    "TD3": TD3Args,
}

def set_args(yaml_path:Optional[str]=None) -> BasicArgs:
    # Load the command line arguments
    cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
    path = cli_cfg.get("yaml_path", yaml_path) # Search from command line first, if not found, use the default path
    cli_cfg.pop("yaml_path", None)
    assert os.path.exists(path), f"YAML file {path} does not exist. Please check and specify the path from command line or arguments."

    # Load the YAML configuration file
    yaml_cfg = OmegaConf.load(path)
    assert "alg_name" in yaml_cfg or "alg_name" in cli_cfg, "Please specify the algorithm name in the yaml file or command line."
    
    # Check the algorithm name and set the corresponding class
    alg_name = yaml_cfg.get("alg_name", cli_cfg.get("alg_name"))
    args_name = next((i for i in ARGS_MAP.keys() if i in alg_name), None)
    ARG_class = ARGS_MAP.get(args_name)
            
    # Set the parameters for the algorithm
    assert ARG_class is not None, f"Algorithm {alg_name} is not supported. Please check the algorithm name in {ARGS_MAP.keys()}."
    default_cfg = OmegaConf.structured(ARG_class)

    # Merge all configurations
    cfg = OmegaConf.merge(default_cfg, yaml_cfg, cli_cfg)
    cfg["alg_name"] = alg_name
    return cfg
    # return OmegaConf.to_object(cfg)

if __name__ == "__main__":
    # Example usage
    config = set_args("recipes/PPO.yaml")
    print(config)