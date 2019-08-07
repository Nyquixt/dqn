import gym
import numpy as np
import time
import argparse

import torch

from atari_wrappers import wrap_deepmind, make_atari

from model import DQN
from utils import to_torch_image

parser = argparse.ArgumentParser()

parser.add_argument('--env', default='PongNoFrameskip-v4', help='Name of the environment')
parser.add_argument('--model', type=str, default='PongNoFrameskip-v4-best.dat', help='Model to be loaded')

args = parser.parse_args()

env = make_atari(args.env)
env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=False)

model = DQN((4, 84, 84), env.action_space.n)
model.load_state_dict(torch.load(args.model))
model.eval()

while True:
    obs = env.reset()
    episode_reward = 0.0
    while True:
        time.sleep(0.01)
        env.render()

        obs = to_torch_image(obs) # convert to torch shape
        state_a = np.array([obs], copy=False).astype(np.float32) / 255.0 # normalize state
        state_v = torch.tensor(state_a)
        q_vals = model(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            print('reward=' + str(episode_reward))
            break