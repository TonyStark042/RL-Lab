from models import MODEL_MAP
from core.args import ARGS_MAP
import gymnasium as gym
from omegaconf import OmegaConf
from gymnasium.wrappers import RecordVideo
from core.args import ARGS_MAP
import os
import numpy as np
import torch
from core import noDeepLearning
from core.baseModule import RL

def make_env(env_name, max_episode_steps=None, record_video=False, video_dir=None):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=max_episode_steps)
        if record_video:
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda ep: ep % 10 == 0,
                name_prefix=env_name,
                fps=60,
            )
        return env
    return _init  # must return a function

def create_agent(alg_name:str, *args, multi_env=True, load=False) -> RL:
    args_name = next((i for i in ARGS_MAP.keys() if i in alg_name), None)
    ARG_class = ARGS_MAP.get(args_name)
    
    # Set the parameters for the algorithm
    default_cfg = OmegaConf.structured(ARG_class)
    default_cfg["alg_name"] = alg_name
    total_args = OmegaConf.merge(default_cfg, *args)
    total_args = OmegaConf.to_object(total_args)

    # create the new agent
    model_class = MODEL_MAP.get(args_name)
    if multi_env:
        env = gym.vector.AsyncVectorEnv([make_env(total_args.env_name, total_args.max_episode_steps) 
                for _ in range(total_args.num_envs)])
    else:
        if load:  # load means in test mode
            env = gym.vector.SyncVectorEnv([make_env(total_args.env_name, total_args.max_episode_steps, record_video=True, video_dir=total_args.save_dir)])
        else:
            env = gym.vector.SyncVectorEnv([make_env(total_args.env_name, total_args.max_episode_steps)])
    agent = model_class(env, total_args)

    # Load the model
    if load:
        save_dir = agent.monitor._check_dir()
        if alg_name in noDeepLearning:
            para = os.path.join(save_dir, "Q_table.npy")
            agent.Q = np.load(para, allow_pickle=True)
        else:
            para = os.path.join(save_dir, "weight.pth")
            models_dict = torch.load(para)
            for net_name, state_dict in models_dict.items():
                model = getattr(agent, net_name)
                model.load_state_dict(state_dict)
        agent.logger.info(f"Loading model from {para}")
        
    return agent