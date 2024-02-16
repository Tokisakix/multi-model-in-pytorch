import numpy as np

from load_config import load_config

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, minimal_size, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.minimal_size = minimal_size
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        return

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        return
    
    def can_sample(self):
        return len(self.buffer) >= self.minimal_size

    def sample(self):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*samples)
        return state, action, reward, next_state, done, indices

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))
        return

    def __len__(self):
        return len(self.buffer)
    
    


# ---TEST---

if __name__ == "__main__":
    BUFFER_CONFIG  = load_config()["buffer"]
    BUFFER_SIZE    = BUFFER_CONFIG["buffer_size"]
    BATCH_SIZE     = BUFFER_CONFIG["batch_size"]
    MINIMAL_SIZE   = BUFFER_CONFIG["minimal_size"]

    buffer = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, MINIMAL_SIZE)

    for idx in range(MINIMAL_SIZE):
        buffer.add([idx], idx, idx, [idx], False)
        if buffer.can_sample():
            break

    state, action, reward, next_state, done, indices = buffer.sample()

    print("state:", state)
    print("action:", action)
    print("reward:", reward)
    print("next_state:", next_state)
    print("done:", done)
    print("indices:", indices)