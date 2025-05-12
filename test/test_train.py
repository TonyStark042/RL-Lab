import os
import sys
import unittest
import gymnasium as gym
import pytest
from unittest.mock import patch
from core import noDeepLearning, discrete_only_algorithms
from utils import make_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MODEL_MAP
from core.args import ARGS_MAP

all_algorithms = list(MODEL_MAP.keys()) + ["DoubleDuelingNoisyDQN"]
continuous_algorithms = [algo for algo in all_algorithms if algo not in discrete_only_algorithms]
all_algorithms.remove("DDPG")
all_algorithms.remove("TD3")

class TestTrain:
    def setup_class(self):
        self.discrete_config = {
            "env_name": "CartPole-v1",
            "max_timesteps": 500,
            "eval_freq": 100,
            "episode_eval_freq": 1,
        }
        self.continuous_config = {
            "env_name": "Pendulum-v1",
            "max_timesteps": 500,
            "eval_freq": 100,
            "episode_eval_freq": 1,
        }

    @pytest.mark.parametrize("alg_name", all_algorithms)
    def test_discrete(self, alg_name):
        if alg_name in noDeepLearning:
            cur_config = self.discrete_config.copy()
            cur_config["env_name"] = "CliffWalking-v0"
            cur_config["max_episode_steps"] = 100
        else:
            cur_config = self.discrete_config
        self._run_train(alg_name, cur_config)

    @pytest.mark.parametrize("alg_name", continuous_algorithms)
    def test_continuous(self, alg_name):
        self._run_train(alg_name, self.continuous_config)

    @pytest.mark.parametrize("alg_name", [
        "A2C",
        "PPO",
    ])
    def test_parallel_training(self, alg_name):
        parallel_config = self.continuous_config.copy()
        parallel_config["num_envs"] = 2
        self._run_train(alg_name, parallel_config)

    def _run_train(self, alg_name, cur_config):
        args_name = alg_name if "DQN" not in alg_name else "DQN"
        save_dir = os.path.join("test/tmp", cur_config["env_name"], alg_name)
        cur_config["save_dir"] = save_dir
        args = ARGS_MAP.get(args_name)(alg_name=alg_name, **cur_config)
        MODEL_CLASS = MODEL_MAP.get(args_name)

        env = gym.vector.AsyncVectorEnv([make_env(args.env_name, args.max_episode_steps) 
            for _ in range(args.num_envs)])
        agent = MODEL_CLASS(env, args)
        agent.train()
        agent.save()
        agent.monitor.learning_curve(mode="timestep")
        if args.episode_eval_freq is not None:
            agent.monitor.learning_curve(mode="episode")
