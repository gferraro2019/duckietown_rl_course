#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using a Logitech Game Controller, as well as record trajectories.
"""

import argparse
import torch

import gym
import numpy as np
import pyglet
from pyglet.window import key

from duckietownrl.gym_duckietown.envs import DuckietownEnv
from duckietownrl.utils.utils import ReplayBuffer
from duckietownrl.utils.wrappers import (
    Wrapper_BW,
    Wrapper_NormalizeImage,
    Wrapper_Resize,
    Wrapper_StackObservation,
)

from duckietownrl.algorithms.sac import SAC

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument(
    "--draw-curve", action="store_true", help="draw the lane following curve"
)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)

parser.add_argument(
    "--max-steps", action="store_true", default=1500, help="number of steps per episode"
)

args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        distortion=args.distortion,
        domain_rand=args.domain_rand,
        max_steps=args.max_steps,
        seed=args.seed,
    )
else:
    env = gym.make(args.env_name)

# wrapping env
n_frames = 4
resize = (48, 64)
env = Wrapper_StackObservation(env, n_frames)
env.append_wrapper(Wrapper_Resize(env, resize=resize))
env.append_wrapper(Wrapper_BW(env))
env.append_wrapper(Wrapper_NormalizeImage(env))


env.reset()
env.render()

# initialize stack
for _ in range(n_frames):
    obs, _, _, _ = env.step([0, 0])

# create replay buffer
batch_size = 64
replay_buffer = ReplayBuffer(10_000, batch_size)

# define an agent
state_dim = (n_frames, *resize)  # Shape of state input (4, 84, 84)
action_dim = 2
# agent = SAC(state_dim, action_dim)
agent = SAC("DuckieTown", state_dim, action_dim, replay_buffer=replay_buffer)
tot_episodes = 0


def update(dt):
    global obs
    global tot_episodes
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    # action = np.random.uniform(low=-1, high=1, size=(2))
    action = agent.select_action(torch.tensor(obs, dtype=torch.float32))

    next_obs, reward, done, info = env.step(action)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, sum(reward)))

    replay_buffer.add(obs, next_obs, action, reward, done)

    # Update the agent if the replay buffer has enough samples
    # if replay_buffer.can_sample():
    #     agent.update(replay_buffer, batch_size=64)
    # if np.random.random() < 0.3:
    agent.train()

    obs = next_obs
    env.render()

    if tot_episodes % 100 == 0:
        agent.save(tot_episodes)

    if True in done:
        print("done!")
        env.reset()
        env.render()
        tot_episodes += 1


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
