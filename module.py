import gymnasium as gym

class RL():
    def __init__(self, env):
        self.env = env
        self.action_num =  sum(self.env.action_space.shape) if type(env.action_space) == gym.spaces.box.Box else self.env.action_space.n
        self.state_num = sum(self.env.observation_space.shape) if type(env.observation_space) == gym.spaces.box.Box else self.env.observation_space.n
        self.max_steps_per_eos = self.env.spec.max_episode_steps