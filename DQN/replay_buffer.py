from random import sample


class Sars:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0

    def insert(self, prev_state, action, reward, next_state, done):
        sars = Sars(prev_state, action, reward, next_state, done)
        self.insert_sars(sars)

    def insert_sars(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        min_val = min(self.idx, self.buffer_size)
        if num_samples > min_val:
            return None
        else:
            return sample(self.buffer[:min_val], num_samples)
