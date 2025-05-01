import gymnasium as gym
from models import MODEL_MAP
from core.args import ARGS_MAP
import pytest
import os
import yaml

@pytest.mark.parametrize("alg_name", [
    "A2C",
    "PPO",
    "REINFORCE",
    "DQN"
])
def test_test(alg_name):
    save_dir = os.path.join("test/tmp", "CartPole-v1", alg_name)
    args_name = alg_name if "DQN" not in alg_name else "DQN"
    para_dict = yaml.safe_load(open(os.path.join(save_dir, "recipe.yaml"), "r"))

    args = ARGS_MAP.get(args_name)(alg_name=alg_name)
    for k, v in para_dict.items():
        if k in args.__dict__:
            setattr(args, k, v)
    MODEL_CLASS = MODEL_MAP.get(args_name)

    env = gym.make(args.env_name, render_mode="rgb_array")
    agent = MODEL_CLASS(env, args)
    agent.test(save_dir)