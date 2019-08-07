from atari_wrappers import make_atari, wrap_deepmind

import time
import argparse
import numpy as np 
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from buffer import ReplayBuffer
from agent import Agent

from tensorboardX import SummaryWriter

"""
    FOR MEMORY SAVING PURPOSE, ONLY CONVERT OBSERVATION TO PYTORCH IMAGE AND NORMALIZE IT IF 
    A NETWORK OPERATION(S) IS INVOLVED. OBSERVATION IS STORED IN LAZYFRAME OBJECT FROM THE WRAPPERS
"""

def calc_loss(batch, net, target_net, gamma=0.99, device="cpu"):
    # states dim should be BATCH x 4 x 84 x 84
    states, actions, rewards, dones, next_states = batch

    # normalize states before feeding to network
    states = states.astype(np.float32) / 255.0
    next_states = next_states.astype(np.float32) / 255.0

    # transform to torch tensors
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # feed through net, Q(s, a)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # feed through target net, Q^(s', a')
    next_state_values = target_net(next_states_v).max(1)[0]

    # set next state vals of terminal state to 0
    next_state_values[done_mask] = 0.0

    # detach() so that gradients won't keep as we freeze the target net
    next_state_values = next_state_values.detach()

    # calculate r + gamma * Q^(s', a')
    expected_state_action_values = next_state_values * gamma + rewards_v

    # find mean squared error as loss
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--env', default='PongNoFrameskip-v4', help='Name of the environment')
    parser.add_argument('--reward', type=float, default=19.5, help='Mean reward bound to stop training')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--target_update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--end_eps', type=float, default=0.02, help='End epsilon')
    parser.add_argument('--frame_eps', type=float, default=100000, help='Frames of epsilon annealing')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--learn_start', type=int, default=10000, help='Learning starts after this number of frames')

    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # make env
    env = make_atari(args.env)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=True, frame_stack=True, scale=False)

    net = DQN((4, 84, 84), env.action_space.n).to(device)
    target_net = DQN((4, 84, 84), env.action_space.n).to(device)
    # copy weights from net to target_net
    target_net.load_state_dict(net.state_dict())

    # summary writer
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    # init stuff
    buffer = ReplayBuffer(args.buffer_size)
    agent = Agent(env, buffer)
    epsilon = 1.0
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    total_rewards = []
    frame_cnt = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    # training loop
    while True:

        epsilon = max(args.end_eps, 1.0 - frame_cnt/args.frame_eps) # will fix this later
        frame_cnt += 1
        reward = agent.play_step(net, epsilon, device)

        if reward is not None: # if terminal state
            total_rewards.append(reward)
            speed = (frame_cnt - ts_frame) / (time.time() - ts)
            ts_frame = frame_cnt
            ts = time.time()
            
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_cnt, len(total_rewards), mean_reward, epsilon, speed))

            # update stats
            writer.add_scalar("epsilon", epsilon, frame_cnt)
            writer.add_scalar("speed", speed, frame_cnt)
            writer.add_scalar("reward_100", mean_reward, frame_cnt)
            writer.add_scalar("reward", reward, frame_cnt)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward

            if mean_reward > args.reward: # if reward exceeds expected reward, stop training
                print("Solved in %d frames!" % frame_cnt)
                break
            
        if len(buffer) < args.learn_start:
            continue

        if frame_cnt % args.target_update == 0: # sync networks
            target_net.load_state_dict(net.state_dict())

        # fit network
        optimizer.zero_grad()
        batch = buffer.sample(args.batch)
        loss_t = calc_loss(batch, net, target_net, 0.99, device)
        loss_t.backward()
        optimizer.step()

    writer.close()
