from collections import deque
import random

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
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
        
    def sample_all(self):
        return zip(*self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)