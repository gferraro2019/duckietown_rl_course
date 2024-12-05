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
    Wrapper_YellowWhiteMask,
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
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument(
    "--draw-curve", action="store_true", help="draw the lane following curve"
)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)

parser.add_argument(
    "--max_steps", default=1500, help="number of steps per episode", type=int
)
parser.add_argument(
    "--return_masked_obs",
    action="store_true",
    default=False,
    help="to use yellow and white wrapper",
)

parser.add_argument(
    "--yellow_mask",
    action="store_true",
    default=False,
    help="to use yellow and white wrapper",
)

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--replay_buffer_size", default=50_000, type=int)
parser.add_argument("--save_on_episode", default=100, type=int)
parser.add_argument("--width_frame", default=28, type=int)
parser.add_argument("--height_frame", default=28, type=int)
parser.add_argument("--width_preview", default=80, type=int)
parser.add_argument("--height_preview", default=60, type=int)

parser.add_argument("--n_chans", default=3, type=int)
parser.add_argument("--n_frames", default=3, type=int)
parser.add_argument("--n_envs", default=1, type=int)
parser.add_argument("--tau", default=0.005, type=float)
parser.add_argument("--reward_invalid_pose", default=-77, type=int)
parser.add_argument("--alpha", default=0.20, type=float)
parser.add_argument("--collect_random_steps", default=3000, type=int)
parser.add_argument("--path", default="/media/g.ferraro/DONNEES", type=str)


args = parser.parse_args()

yellow_mask = args.yellow_mask
n_frames = args.n_frames
n_chans = args.n_chans  # 1 for B/W images 3 for RGBs
n_envs = args.n_envs
resize_shape = (args.width_frame, args.height_frame)  # (width,height)
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
        window_width=args.width_preview,
        window_height=args.height_preview,
        camera_width=resize_shape[0],
        camera_height=resize_shape[1],
        reward_invalid_pose=args.reward_invalid_pose,
        # user_tile_start=(2, 0),
        # start_pose=(0.34220727, 0, 0.58371305),
        # start_angle=np.pi / 2,
    )
    k += 1
    # wrapping the environment
    env = Wrapper_StackObservation(env, n_frames, n_chans=n_chans)
    if n_chans == 1:
        env.append_wrapper(Wrapper_BW(env))
    env.append_wrapper(Wrapper_NormalizeImage(env))

    if yellow_mask:
        return_mask = args.return_masked_obs
        env.append_wrapper(Wrapper_YellowWhiteMask(env, return_mask))
        if return_mask is False:
            n_chans += 3
            env.n_chans += 3

    env.reset()
    env.render(mode="rgb_array")
    envs.append(env)


device = "cuda"

# assemble first obervation
l_obs = []
obs, _, _, _ = env.step(np.array([0, 0], dtype=np.float32))

for _ in envs:
    l_obs.append(np.asarray(obs))
obs = np.stack(l_obs, axis=0)

# create replay buffer
batch_size = args.batch_size
state_dim = (n_frames * n_chans, *resize_shape)  # Shape of state input (4, 84, 84)
action_dim = 2
replay_buffer = ReplayBuffer(
    args.replay_buffer_size,
    batch_size,
    state_dim,
    action_dim,
    normalize_rewards=False,
    device=device,
)
# replay_buffer = load_replay_buffer()

# define an agent
agent = SAC(
    "DuckieTown",
    state_dim,  # envs[0].observation_space.shape[:3],
    envs[0].action_space.shape[0],
    replay_buffer=replay_buffer,
    device=device,
    actor_lr=0.001,
    tau=args.tau,
    alpha=args.alpha,
)

tot_episodes = 0
timesteps = 0
probability_training = 1.0
save_on_episodes = args.save_on_episode
running_avg_reward = 0

folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
path = args.path

eps_returns = np.zeros(n_envs)
once = True
collect_random_timesteps = args.collect_random_steps


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    global running_avg_reward
    global eps_returns
    global once
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    next_observations = []
    rewards = []
    dones = []

    if timesteps < collect_random_timesteps:
        action = 2 * np.random.rand(n_envs, 2) - 1

    else:
        action = agent.select_action(torch.tensor(obs, dtype=torch.float32).to(device))

    for i, env in enumerate(envs):
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
    if (
        np.random.random() < probability_training
        and timesteps >= collect_random_timesteps
    ):
        agent.train(timesteps, device)

    obs = next_obs

    if tot_episodes > 0 and tot_episodes % save_on_episodes == 0 and once:
        agent.save(path, folder_name, tot_episodes)
        once = False

    if True in done:
        idx = np.where(done)[0]
        idx_envs = idx % n_envs
        for id in idx_envs:
            env = envs[id]
            print(f"env N.{id} done!")
            wandb.log({"ep_return": eps_returns[id], "step_count": env.step_count})
            obs_env = env.reset()
            env.render()
            obs[id] = np.stack(obs_env[0], axis=0)

            # env.render()
            tot_episodes += 1
            eps_returns[id] = 0.0
            once = True

    for env in envs[-4:]:
        env.render(mode="human")

    timesteps += 1


dt = 0.001
t = time.time()
while True:
    update(dt)
