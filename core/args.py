from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass(kw_only=True, frozen=False)
class Args:
    max_epochs:int = np.inf
    max_timesteps:int = np.inf
    reward_threshold:float = None
    early_stop:bool = True
    baseline:float = 0
    gamma:float = 0.99
    lr:float = 1e-4
    h_size:int = 32
    window_size:int = 10
    timestep_freq:int = None
    report_freq:int = 100
    custom_args: dict = field(default_factory=dict)

    def __post_init__(self):
        pass

@dataclass(kw_only=True, frozen=False)
class VRLArgs(Args):
    epsilon_start:float = 1.0
    epsilon_end:float = 0.01
    epsilon_decay:float = 0.002
    epsilon_decay_flag:bool = True
    sync_steps:int = 64, 
    batch_size:int = 32, 
    memory_size:int = 6000,  

@dataclass(kw_only=True, frozen=False)
class PRLArgs(Args):
    is_gae:bool = False
    lmbda:float = 0.95