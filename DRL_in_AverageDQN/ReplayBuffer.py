import collections
import random

from load_config import load_config

class ReplayBuffer:
    def __init__(self, capacity, batch_size, minimal_size):
        self.buffer       = collections.deque(maxlen=capacity)
        self.batch_size   = batch_size
        self.minimal_size = minimal_size
        return

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        return
    
    def can_sample(self):
        return len(self.buffer) >= self.minimal_size

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)
    
    


# ---TEST---

if __name__ == "__main__":
    BUFFER_CONFIG  = load_config()["buffer"]
    BUFFER_SIZE    = BUFFER_CONFIG["buffer_size"]
    BATCH_SIZE     = BUFFER_CONFIG["batch_size"]
    MINIMAL_SIZE   = BUFFER_CONFIG["minimal_size"]

    buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, MINIMAL_SIZE)

    for idx in range(MINIMAL_SIZE):
        buffer.add([idx], idx, idx, [idx], False)
        if buffer.can_sample():
            break

    state, action, reward, next_state, done = buffer.sample()

    print("state:", state)
    print("action:", action)
    print("reward:", reward)
    print("next_state:", next_state)
    print("done:", done)