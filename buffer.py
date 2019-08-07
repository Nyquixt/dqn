import numpy as np 
import collections

from utils import to_torch_image_batch

Transition = collections.namedtuple('Transition', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return to_torch_image_batch(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), to_torch_image_batch(next_states)