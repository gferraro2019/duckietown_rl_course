#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using a Logitech Game Controller, as well as record trajectories.
"""

import argparse
import json
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument(
    "--draw-curve", action="store_true", help="draw the lane following curve"
)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        distortion=args.distortion,
        domain_rand=args.domain_rand,
        max_steps=np.inf,
    )
else:
    env = gym.make(args.env_name)


import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from src.algorithms.ddpg import DDPG
from utils import seed, evaluate_policy, ReplayBuffer
from src.utils.env import launch_env
from src.utils.wrappers import (
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
)


def _train(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized DDPG")

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    print("Starting training")
    while total_timesteps < args.max_timesteps:

        print("timestep: {} | reward: {}".format(total_timesteps, reward))

        if done:
            if total_timesteps != 0:
                print(
                    ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                    % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                )
                policy.train(
                    replay_buffer,
                    episode_timesteps,
                    args.batch_size,
                    args.discount,
                    args.tau,
                )

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print(
                        "rewards at time {}: {}".format(
                            total_timesteps, evaluations[-1]
                        )
                    )

                    if args.save_models:
                        policy.save(file_name="ddpg", directory=args.model_dir)
                    np.savez("./results/rewards.npz", evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (
                    action
                    + np.random.normal(
                        0, args.expl_noise, size=env.action_space.shape[0]
                    )
                ).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    print("Training done, about to save..")
    policy.save(filename="ddpg", directory=args.model_dir)
    print("Finished saving..should return now!")


# DDPG Args
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument(
    "--start_timesteps", default=1e4, type=int
)  # How many time steps purely random policy is run for
parser.add_argument(
    "--eval_freq", default=5e3, type=float
)  # How often (time steps) we evaluate
parser.add_argument(
    "--max_timesteps", default=1e6, type=float
)  # Max time steps to run environment for
parser.add_argument(
    "--save_models", action="store_true", default=True
)  # Whether or not models are saved
parser.add_argument(
    "--expl_noise", default=0.1, type=float
)  # Std of Gaussian exploration noise
parser.add_argument(
    "--batch_size", default=32, type=int
)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument(
    "--policy_noise", default=0.2, type=float
)  # Noise added to target policy during critic update
parser.add_argument(
    "--noise_clip", default=0.5, type=float
)  # Range to clip target policy noise
parser.add_argument(
    "--policy_freq", default=2, type=int
)  # Frequency of delayed policy updates
parser.add_argument(
    "--env_timesteps", default=500, type=int
)  # Frequency of delayed policy updates
parser.add_argument(
    "--replay_buffer_max_size", default=10000, type=int
)  # Maximum number of steps to keep in the replay buffer
parser.add_argument("--model-dir", type=str, default="reinforcement/pytorch/models/")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

env.reset()
env.render()
_train(parser.parse_args())

# # global variables for demo recording
# positions = []
# actions = []
# demos = []
# recording = False


# def write_to_file(demos):
#     num_steps = 0
#     for demo in demos:
#         num_steps += len(demo["actions"])
#     print("num demos:", len(demos))
#     print("num steps:", num_steps)

#     # Store the trajectories in a JSON file
#     with open("experiments/demos_{}.json".format(args.map_name), "w") as outfile:
#         json.dump({"demos": demos}, outfile)


# def process_recording():
#     global positions, actions, demos

#     if len(positions) == 0:
#         # Nothing to delete
#         if len(demos) == 0:
#             return

#         # Remove the last recorded demo
#         demos.pop()
#         write_to_file(demos)
#         return

#     p = list(map(lambda p: [p[0].tolist(), p[1]], positions))
#     a = list(map(lambda a: a.tolist(), actions))

#     demo = {"positions": p, "actions": a}

#     demos.append(demo)

#     # Write all demos to this moment
#     write_to_file(demos)


# @env.unwrapped.window.event
# def on_key_press(symbol, modifiers):
#     """
#     This handler processes keyboard commands that
#     control the simulation
#     """

#     if symbol == key.BACKSPACE or symbol == key.SLASH:
#         print("RESET")
#         env.reset()
#         env.render()
#     elif symbol == key.PAGEUP:
#         env.unwrapped.cam_angle[0] = 0
#         env.render()
#     elif symbol == key.ESCAPE:
#         env.close()
#         sys.exit(0)


# @env.unwrapped.window.event
# def on_joybutton_press(joystick, button):
#     """
#     Event Handler for Controller Button Inputs
#     Relevant Button Definitions:
#     1 - A - Starts / Stops Recording
#     0 - X - Deletes last Recording
#     2 - Y - Resets Env.

#     Triggers on button presses to control recording capabilities
#     """
#     global recording, positions, actions

#     # A Button
#     if button == 1:
#         if not recording:
#             print("Start recording, Press A again to finish")
#             recording = True
#         else:
#             recording = False
#             process_recording()
#             positions = []
#             actions = []
#             print("Saved recording")

#     # X Button
#     elif button == 0:
#         recording = False
#         positions = []
#         actions = []
#         process_recording()
#         print("Deleted last recording")

#     # Y Button
#     elif button == 3:
#         print("RESET")
#         env.reset()
#         env.render()

#     # Any other button thats not boost prints help
#     elif button != 5:
#         helpstr1 = "A - Starts / Stops Recording\nX - Deletes last Recording\n"
#         helpstr2 = "Y - Resets Env.\nRB - Hold for Boost"

#         print("Help:\n{}{}".format(helpstr1, helpstr2))


# def update(dt):
#     """
#     This function is called at every frame to handle
#     movement/stepping and redrawing
#     """
#     global recording, positions, actions

#     # No actions took place
#     if round(joystick.x, 2) == 0.0 and round(joystick.y, 2) == 0.0:
#         return

#     x = round(joystick.y, 2)
#     z = round(joystick.x, 2)

#     action = np.array([-x, -z])

#     # Right trigger, speed boost
#     if joystick.buttons[5]:
#         action *= 1.5

#     if recording:
#         positions.append((env.unwrapped.cur_pos, env.unwrapped.cur_angle))
#         actions.append(action)

#     # Giuseppe's modification for going forward while turning
#     if action[0] == 0 and (action[1] == 1 or action[1] == -1):
#         action[0] = 0.2
#         action[1] *= 1.5

#     obs, reward, done, info = env.step(action)
#     print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

#     if done:
#         print("done!")
#         env.reset()
#         env.render()

#         if recording:
#             process_recording()
#             positions = []
#             actions = []
#             print("Saved Recoding")

#     env.render()


# pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# # Registers joysticks and recording controls
# joysticks = pyglet.input.get_joysticks()
# assert joysticks, "No joystick device is connected"
# joystick = joysticks[0]
# joystick.open()
# joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()

env.close()
