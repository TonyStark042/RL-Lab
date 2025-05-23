from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity:int=10000) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self,transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size:int, sequential:bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)
        batch = map(np.array, list(zip(*batch)))
        return batch
        
    def sample_all(self, clear:bool = True):
        if clear:
            batch = self.buffer.copy()
            self.clear()
        else:
            batch = self.buffer
        batch = map(np.array, list(zip(*batch)))
        return batch

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer) * (self.buffer[0][0].shape[0] if self.buffer[0][0].ndim > 1 else 1)

class HorizonBuffer(ReplayBuffer):
    def __init__(self, horizon, capacity:int=10000) -> None:
        super().__init__(capacity)
        self.horizon = horizon

    def is_ready(self, **kwargs):
        timestep = kwargs.get("timestep", None)
        if timestep % self.horizon == 0 and len(self.buffer) > 0:
            return True
        else:
            return False

class EpisodeBuffer(ReplayBuffer):
    def __init__(self, capacity:int=10000) -> None:
        super().__init__(capacity)

    def is_ready(self, **kwargs):
        done = kwargs.get("done", None)
        if done:
            return True
        else:
            return False
        