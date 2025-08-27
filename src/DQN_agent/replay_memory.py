import random
from collections import deque, namedtuple
from typing import Deque

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class replay_memory:
    def __init__(self, capacity):
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
