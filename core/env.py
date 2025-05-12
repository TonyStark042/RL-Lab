from gymnasium.vector import AsyncVectorEnv
from typing import Optional
import numpy as np
from gymnasium.spaces import Box, Discrete

class WrappedEnv(AsyncVectorEnv):
    def __init__(self, env: AsyncVectorEnv):
        self._env = env
        self.action_dim: int = 0
        self.max_action: Optional[float] = None
        self.action_num: Optional[float] = None
        self.state_dim: int = 0
        self.state_num: Optional[float] = None
        self.has_continuous_action_space: bool = False
        self._setup_env_properties()
        
    def _setup_env_properties(self):
        if isinstance(self._env.action_space, Box):
            self.has_continuous_action_space = True
        elif isinstance(self._env.action_space, Discrete):
            self.has_continuous_action_space = False

        if self.has_continuous_action_space:
            assert np.allclose(np.abs(self._env.action_space.low), np.abs(self._env.action_space.high)), \
                "Continuous action space must be symmetric"
            self.max_action = self._env.single_action_space.high[0]
            self.action_dim = self._env.single_action_space.shape[0]
            self.action_num = np.inf
        else:
            self.action_dim = 1
            self.action_num = self._env.single_action_space.n

        self.state_dim = self._env.single_observation_space.shape[0] if len(self._env.single_observation_space.shape) != 0 else 1
        self.state_num = self._env.single_observation_space.n if hasattr(self._env.single_observation_space, "n") else np.inf
    
    @property
    def spec(self):
        return self._env.spec

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def step(self, action):
        return self._env.step(action)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)