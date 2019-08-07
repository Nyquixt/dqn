import numpy as np
import torch

from buffer import Transition
from utils import to_torch_image

class Agent:
    def __init__(self, env, exp_replay):
        self.env = env
        self.exp_replay = exp_replay
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # choose action based on epsilon-greedy
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = to_torch_image(self.state) # convert to torch shape
            state_a = np.array([state], copy=False).astype(np.float32) / 255.0 # normalize state
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)

        # update reward
        self.total_reward += reward

        # push new transition to the replay buffer
        exp = Transition(self.state, action, reward, is_done, new_state)
        self.exp_replay.append(exp)
        
        # set current state to new state
        self.state = new_state

        if is_done: # terminal state
            # retrieve total reward in the episode
            done_reward = self.total_reward
            # reset the environment
            self._reset()

        return done_reward