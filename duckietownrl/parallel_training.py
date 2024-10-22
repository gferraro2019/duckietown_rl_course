#!/usr/bin/env python3
import argparse
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


n_frames = 4
n_envs = 10
resize_shape = (32, 24)  # (width,height)
envs = []
k = 0
for _ in range(n_envs):
    env = DuckietownEnv(
        map_name=args.map_name,
        distortion=args.distortion,
        domain_rand=args.domain_rand,
        max_steps=args.max_steps,
        seed=args.seed + k,
    )
    k += 1
    # wrapping the environment
    env = Wrapper_StackObservation(env, n_frames)
    env.append_wrapper(Wrapper_Resize(env, shape=resize_shape))
    env.append_wrapper(Wrapper_BW(env))
    env.append_wrapper(Wrapper_NormalizeImage(env))

    env.reset()
    env.render(mode="rgb_array")
    envs.append(env)


# assemble first obervation
l_obs = []
obs, _, _, _ = env.step(np.array([0, 0], dtype=np.float32))

for _ in envs:
    l_obs.append(obs)
obs = np.stack(l_obs, axis=0)

# create replay buffer
batch_size = 256
replay_buffer = ReplayBuffer(
    n_envs * 26, batch_size, normalize_rewards=False, device="cpu"
)
# replay_buffer = load_replay_buffer()

# define an agent
state_dim = (n_frames, *resize_shape)  # Shape of state input (4, 84, 84)
action_dim = 2
agent = SAC(
    "DuckieTown",
    state_dim,  # envs[0].observation_space.shape[:3],
    envs[0].action_space.shape[0],
    replay_buffer=replay_buffer,
    device="cpu",
)
tot_episodes = 0
timesteps = 0
probability_training = 0.66
save_on_episodes = 200

folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
path = "/media/g.ferraro/DONNEES"


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    next_observations = []
    rewards = []
    dones = []

    action = agent.select_action(torch.tensor(obs, dtype=torch.float32))
    # noise = np.random.randint(-300, 300, (n_envs, 2)) * 0.0001
    # noisy_action = action + noise

    # add noise to actions
    for i, env in enumerate(envs):
        # next_obs, reward, done, info = env.step(noisy_action[i])
        next_obs, reward, done, info = env.step(action[i])
        next_observations.append(next_obs)
        rewards.append(reward)
        dones.append(done)

    next_obs = np.stack(next_observations, axis=0)
    reward = np.stack(rewards, axis=0)
    done = np.stack(dones, axis=0)
    # action = noisy_action

    next_observations.clear()
    rewards.clear()
    dones.clear()

    print(
        f"eps = {tot_episodes} step_count = {timesteps}, rewards={reward.sum(0)/n_envs}"
    )

    replay_buffer.add(obs, next_obs, action, reward, done)

    # Train with a certain probability for computing efficiency
    if np.random.random() < probability_training:
        agent.train()

    obs = next_obs

    for env in envs[-2:]:
        env.render(mode="human")

    if tot_episodes > 0 and tot_episodes % save_on_episodes == 0:
        agent.save(path, folder_name, tot_episodes)

    if True in done:
        idx = np.where(np.any(done, axis=1))[0]
        idx_envs = idx % n_envs
        for id in idx_envs:
            env = envs[id]
            print(f"env N.{id} done!")
            obs_env = env.reset()
            obs[id] = np.stack(obs_env[0], axis=0)
            # env.render()
            tot_episodes += 1

    timesteps += 1


dt = 0.001
t = time.time()
while True:
    if time.time() - t > dt:
        update(dt)
        t = time.time()
