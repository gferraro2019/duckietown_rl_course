import random

import gym
import numpy as np
import torch
import os
import pickle

from zmq import device


def load_replay_buffer(filename="replay_buffer"):
    print("loading replay buffer...")
    file = open(filename, "rb")
    replay_buffer = pickle.load(file)
    file.close()
    print("Done")
    print(len(replay_buffer))
    return replay_buffer


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


class ReplayBuffer:

    @torch.no_grad()
    def __init__(
        self,
        capacity=10_000,
        batch_size=32,
        state_shape=(4, 120, 160),
        action_shape=(1, 2),
        device="cpu",
        normalize_rewards=False,
    ):
        self.device = device
        # self.content = []
        self.state_shape = state_shape

        self.capacity = capacity
        self.idx = 0
        self.filled = False
        self.batch_size = batch_size
        self.indices = np.zeros(batch_size)
        self.normalize_rewards = normalize_rewards
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.states_img = (
            torch.empty((capacity, *state_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.actions = (
            torch.empty((capacity, action_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.rewards = (
            torch.empty(capacity, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img = (
            torch.empty((capacity, *state_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.dones = torch.empty(capacity, dtype=torch.bool).detach().to(self.device)

    def change_bacth_size(self, batch_size):
        self.batch_size = batch_size
        del self.indices
        self.indices = np.zeros(batch_size)

    def save(self, filename="replay_buffer"):
        print("saving replay buffer...")
        file = open(filename, "wb")
        pickle.dump(self, file, 4)
        file.close()
        print("Done")

    @torch.no_grad()
    def add(self, obs, next_obs, actions, rewards, dones):
        temp_tensor = torch.tensor(obs, dtype=torch.float32).detach().to(self.device)
        size_sample = temp_tensor.shape[0]
        interval = [self.idx, 0]
        if self.idx + size_sample < self.capacity:
            interval[1] = self.idx + size_sample
        else:
            interval[1] = None  # size_sample + interval[0]

        self.states_img[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(actions, dtype=torch.float32).detach().to(self.device)
        )
        self.actions[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(rewards, dtype=torch.float32).detach().to(self.device)
        )
        self.rewards[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(next_obs, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = torch.tensor(dones, dtype=torch.float32).detach().to(self.device)
        self.dones[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        if self.idx + size_sample == self.capacity:
            self.filled = True

        self.idx = (self.idx + size_sample) % self.capacity

    def can_sample(self):
        res = False
        # if len(self) >= self.capacity:
        if len(self) >= self.batch_size * 2:
            res = True
        # print(f"{len(self)} collected")
        return res

    @torch.no_grad()
    def sample(self, sample_capacity=None, device="cpu"):
        if self.can_sample():
            if sample_capacity:
                idx = random.sample(range(len(self)), sample_capacity)

            else:
                idx = random.sample(range(len(self)), self.batch_size)
            self.indices[:] = idx

            rewards = self.rewards[self.indices].to(device)

            if self.normalize_rewards is True:
                rewards = (rewards - self.rewards.min()) / (
                    self.rewards.max() - self.rewards.min()
                )
            return (
                self.states_img[self.indices].to(device),
                self.actions[self.indices].to(device),
                rewards,
                self.next_states_img[self.indices].to(device),
                self.dones[self.indices].to(device),
            )
        else:
            assert "Can't sample: not enough elements!"

    def __len__(self):
        size = 0
        if self.filled:
            size = self.dones.shape[0]
        else:
            size = self.idx
        return size

    @torch.no_grad()
    def increase_capacity(self, new_capacity):
        assert new_capacity > self.capacity
        increment = new_capacity - self.capacity
        self.states_img = torch.cat(
            (
                self.states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        )
        self.actions = torch.cat(
            (
                self.actions,
                torch.empty((increment, self.action_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.rewards = torch.cat(
            (
                self.rewards,
                torch.empty(increment, dtype=torch.float32).detach().to(self.device),
            )
        ).to(self.device)

        self.next_states_img = torch.cat(
            (
                self.next_states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.dones = torch.cat(
            (
                self.dones,
                torch.empty(increment, dtype=torch.bool).detach().to(self.device),
            )
        ).to(self.device)
        self.filled = False
        self.idx = self.capacity
        self.capacity = new_capacity


"""
# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size, batch_size):
        self.storage = []
        self.max_size = max_size
        self.batch_size = batch_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))

    def sample(self, flat=False):
        ind = np.random.randint(0, len(self.storage), size=self.batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1, 1),
            "done": np.stack(dones).reshape(-1, 1),
        }

    def __len__(self):
        return len(self.storage)
        """


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward
