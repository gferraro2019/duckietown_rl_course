#!/usr/bin/env python3
import argparse
from sympy import false
import torch
from datetime import datetime
import os
import time

import gym
import pickle
import numpy as np
import pyglet
from pyglet.window import key  # do not remove, otherwhise render issue


from duckietownrl.gym_duckietown.envs import DuckietownEnv
from duckietownrl.utils.utils import ReplayBuffer, load_replay_buffer
from duckietownrl.utils.wrappers import (
    Wrapper_BW,
    Wrapper_NormalizeImage,
    Wrapper_Resize,
    Wrapper_StackObservation,
)

from duckietownrl.algorithms.sac_new_2dconv import SAC

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="duckietownrl-sac-conv2d",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "duckietown-simulator",
        "epochs": 0,
    },
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="small_loop_bordered")
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


n_frames = 3
n_envs = 1
resize_shape = (28, 28)  # (width,height)
envs = []
k = 0
for i in range(n_envs):
    print(f"creating env N.{i}...")
    env = DuckietownEnv(
        map_name=args.map_name,
        distortion=args.distortion,
        domain_rand=args.domain_rand,
        max_steps=args.max_steps,
        seed=args.seed + k,
        window_width=600,
        window_height=600,
        camera_width=resize_shape[0],
        camera_height=resize_shape[1],
    )
    k += 1
    # wrapping the environment
    env = Wrapper_StackObservation(env, n_frames)
    # env.append_wrapper(Wrapper_Resize(env, shape=resize_shape))
    env.append_wrapper(Wrapper_BW(env))
    env.append_wrapper(Wrapper_NormalizeImage(env))

    env.reset()
    env.render(mode="human")
    envs.append(env)


device = "cuda"

# assemble first obervation
l_obs = []
obs, _, _, _ = env.step(np.array([0, 0], dtype=np.float32))

for _ in envs:
    l_obs.append(obs)
obs = np.stack(l_obs, axis=0)

# create replay buffer
batch_size = 64
state_dim = (n_frames, *resize_shape)  # Shape of state input (4, 84, 84)
action_dim = 2
replay_buffer = ReplayBuffer(
    500_000, batch_size, state_dim, action_dim, normalize_rewards=False, device=device
)
# replay_buffer = load_replay_buffer()

# define an agent
agent = SAC(
    "DuckieTown",
    state_dim,  # envs[0].observation_space.shape[:3],
    envs[0].action_space.shape[0],
    replay_buffer=replay_buffer,
    device=device,
)
tot_episodes = 0
timesteps = 0
probability_training = 1.0
save_on_episodes = 200
running_avg_reward = 0

folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
path = "/media/g.ferraro/DONNEES"

eps_returns = np.zeros(n_envs)


def action_from_joystick():
    x = round(joystick.y, 2)
    z = round(joystick.x, 2)
    y = round(joystick.z, 2)

    action = np.array([-x, -z])
    # Giuseppe's modification for going forward while turning
    # if action[0] == 0 and (action[1] == 1 or action[1] == -1):
    #     action[0] = 0.2
    #     action[1] *= 1.5
    # print([x, y, z])
    return np.expand_dims(action, axis=0)


stop_collecting = False
import sys


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global stop_collecting

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
        env.render()
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.J:
        stop_collecting = not stop_collecting


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    global running_avg_reward
    global eps_returns
    global stop_collecting
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    next_observations = []
    rewards = []
    dones = []

    if stop_collecting is not True:
        # No actions took place
        if abs(round(joystick.x, 2)) <= 0.1 and abs(round(joystick.y, 2)) <= 0.1:
            return
        else:
            action = action_from_joystick()
    else:
        action = agent.select_action(torch.tensor(obs, dtype=torch.float32).to(device))
    # noise = np.random.randint(-300, 300, (n_envs, 2)) * 0.0001
    # noisy_action = action + noise

    # add noise to actions
    for i, env in enumerate(envs):
        # next_obs, reward, done, info = env.step(noisy_action[i])
        next_obs, reward, done, info = env.step(action[i])
        next_observations.append(next_obs)
        rewards.append(reward)
        dones.append(done)
        wandb.log(
            {
                "actor_loss": agent.actor_loss_value,
                "q_loss": agent.q_loss_value,
                "speed": env.speed,
                "lp.dot_dir": env.lp.dot_dir,
                "lp.dist": env.lp.dist,
            }
        )

    next_obs = np.stack(next_observations, axis=0)
    reward = np.stack(rewards, axis=0)
    done = np.stack(dones, axis=0)
    # action = noisy_action

    next_observations.clear()
    rewards.clear()
    dones.clear()

    avg_reward = reward.sum(0) / n_envs
    running_avg_reward += (avg_reward - running_avg_reward) / (timesteps + 1)
    print(
        f"eps = {tot_episodes} step_count = {timesteps}, avg_reward={avg_reward:.3f}, runn_avg_reward={running_avg_reward:.3f}"
    )

    wandb.log(
        {
            "avg_reward": avg_reward,
            "runn_avg_reward": running_avg_reward,
        }
    )

    eps_returns += reward
    replay_buffer.add(obs, next_obs, action, reward, done)

    # Train with a certain probability for computing efficiency
    if np.random.random() < probability_training:
        agent.train(timesteps, device)

    obs = next_obs

    for env in envs[-4:]:
        env.render(mode="human")

    if tot_episodes > 0 and tot_episodes % save_on_episodes == 0:
        agent.save(path, folder_name, tot_episodes)

    if True in done:
        idx = np.where(done)[0]
        idx_envs = idx % n_envs
        for id in idx_envs:
            env = envs[id]
            print(f"env N.{id} done!")
            wandb.log({"ep_return": eps_returns[id], "step_count": env.step_count})
            obs_env = env.reset()
            obs[id] = np.stack(obs_env[0], axis=0)

            # env.render()
            tot_episodes += 1
            eps_returns[id] = 0.0

    timesteps += 1


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Registers joysticks and recording controls
joysticks = pyglet.input.get_joysticks()
assert joysticks, "No joystick device is connected"
joystick = joysticks[0]
joystick.open()
# joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()


dt = 0.001
t = time.time()
while True:
    # if time.time() - t > dt:
    update(dt)
    # t = time.time()
