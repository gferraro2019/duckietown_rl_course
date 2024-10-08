#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using a Logitech Game Controller, as well as record trajectories.
"""

import argparse

import gym
import numpy as np
import pyglet
from pyglet.window import key

from duckietownrl.gym_duckietown.envs import DuckietownEnv
from duckietownrl.utils.utils import ReplayBuffer
from duckietownrl.utils.wrappers import Wrapper_BW


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
env = Wrapper_BW(env)

obs = env.reset()
env.render()


replay_buffer = ReplayBuffer(10_000)


def update(dt):
    global obs
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.random.uniform(low=-1, high=1, size=(2))

    next_obs, reward, done, info = env.step(action)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

    if done:
        print("done!")
        env.reset()
        env.render()

    replay_buffer.add(obs, next_obs, action, reward, done)
    obs = next_obs
    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
