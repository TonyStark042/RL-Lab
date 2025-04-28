import torch
import sys
import numpy as np
from typing import Literal
import logging
from tqdm import tqdm

class Normalizer:
    """
    Normalizer class for normalizing state and reward.
    """
    def __init__(self, dim, clip=10):
        self.clip = clip
        self.count = 0
        self.dim = dim
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        name = "reward_normalizer" if dim == 1 else "state_normalizer"
        self.logger = logging.getLogger(name=name)

    @classmethod
    def init_normalizer(cls, env, mode:Literal["state", "reward"], epochs=10):
        """
        Initialize the normalizer by randomly sampling from the environment.
        """
        samples = []
        if mode == "state":
            dim = env.observation_space.shape[0]
        elif mode == "reward":
            dim = 1

        normalizer = cls(dim=dim, clip=10)

        for epoch in tqdm(range(epochs), desc=f"Initializing {mode}_normalizer"):
            obs = env.reset()
            while True:
                action = env.action_space.sample()  # random explore
                obs, reward, teminate, truncate, info = env.step(action)
                if mode == "state":
                    samples.append(obs)
                elif mode == "reward":
                    samples.append(reward)
                if teminate or truncate:
                    break
        else:
            env.reset()
        normalizer.update(np.array(samples))
        normalizer.logger.info(f"{mode}_normalizer have been initialized, mean: {normalizer.mean}, var: {normalizer.var}")
        return normalizer

    @classmethod
    def loading_normalizer(cls, mean: np.ndarray, var: np.ndarray, count: int, clip=10):
        normalizer = cls(dim=mean.shape[0], clip=clip)
        normalizer.mean = mean
        normalizer.var = var
        normalizer.count = count
        return normalizer

    def normalize(self, x):
        """
        Normalize the input values.
        """
        std = np.maximum(np.sqrt(self.var), 1e-6)
        normalized_x = (x - self.mean) / std
        return np.clip(normalized_x, -self.clip, self.clip)
    
    def update(self, batch_values):
        """
        Update the normalizer with new batch values.
        """
        batch_values = np.array(batch_values).reshape(-1, self.dim)
        batch_count =  batch_values.shape[0]
        batch_mean = np.mean(batch_values, axis=0)
        batch_var = np.var(batch_values, axis=0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count) # Pooled Variance Formula
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
    
    def __call__(self, x):
        return self.normalize(x)

def get_name():
    for i, arg in enumerate(sys.argv):
        if arg == '--alg_name':
            if i + 1 < len(sys.argv):
                alg_name = sys.argv[i + 1]
                break
    else:
        raise ValueError("Algorithm name not found in command line arguments.")
    return alg_name

