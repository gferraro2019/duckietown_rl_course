import gym
from gym import spaces
import numpy as np

from duckietownrl.gym_duckietown.envs.duckietown_env import DuckietownEnv
from duckietownrl.gym_duckietown.simulator import Simulator


class MotionBlurWrapper(Simulator):
    def __init__(self, env=None):
        Simulator.__init__(self)
        self.env = env
        self.frame_skip = 3
        self.env.delta_time = self.env.delta_time / self.frame_skip

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        motion_blur_window = []
        for _ in range(self.frame_skip):
            obs = self.env.render_obs()
            motion_blur_window.append(obs)
            self.env.update_physics(action)

        # Generate the current camera image

        obs = self.env.render_obs()
        motion_blur_window.append(obs)
        obs = np.average(motion_blur_window, axis=0, weights=[0.8, 0.15, 0.04, 0.01])

        misc = self.env.get_agent_info()

        d = self.env._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype,
        )
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize

        return imresize(observation, self.shape)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


import cv2


class Wrapper(DuckietownEnv):
    def __init__(self, env):
        keys_to_keep = ["map_name", "distortion", "domain_rand", "max_steps", "seed"]
        env_to_dict = vars(env)
        kwargs = {k: env_to_dict[k] for k in keys_to_keep if k in env_to_dict}
        self.wrappers_list = []
        super().__init__(**kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def append_wrapper(self, wrapper):
        self.wrappers_list.append(wrapper)
        self.observation_space = wrapper.observation_space

    def reset(self):
        obs = super(DuckietownEnv, self).reset()
        return self.apply_transformation(obs)

    def apply_transformation(self, img):
        for wrapper in self.wrappers_list:
            img = wrapper.apply_transformation(img)
        return img

    def step(self, action):
        obs = super(DuckietownEnv, self).step(action)
        img = obs[0]
        img = self.apply_transformation(img)
        return (img, *obs[1:])


class Wrapper_BW(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(env.observation_space.shape) == 4:
            n_imgs, img_height, img_width, n_channels = env.observation_space.shape
            self.observation_space = spaces.Box(
                0,
                255,
                (n_imgs, img_height, img_width, 1),
                dtype=self.observation_space.dtype,
            )
        else:
            img_height, img_width, n_channels = env.observation_space.shape
            self.observation_space = spaces.Box(
                0,
                255,
                (img_height, img_width, 1),
                dtype=self.observation_space.dtype,
            )

    def apply_transformation(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class Wrapper_Resize(Wrapper):
    def __init__(self, env, shape):
        self.shape = shape
        super().__init__(env)
        if len(env.observation_space.shape) == 4:
            n_imgs, img_height, img_width, n_channels = env.observation_space.shape
            self.observation_space = spaces.Box(
                0,
                255,
                (n_imgs, shape[-1], shape[-2], n_channels),
                dtype=self.observation_space.dtype,
            )
        else:
            img_height, img_width, n_channels = env.observation_space.shape
            self.observation_space = spaces.Box(
                0,
                255,
                (shape[-1], shape[-2], n_channels),
                dtype=self.observation_space.dtype,
            )

    def apply_transformation(self, img):
        return cv2.resize(img, self.shape)


class Wrapper_NormalizeImage(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def apply_transformation(self, img):
        return img / 255.0


class Wrapper_StackObservation(Wrapper):
    def __init__(self, env, n_obs):
        self.n_obs = n_obs
        self.observations_stack = []
        self.rewards_stack = []
        self.dones_stack = []
        self.infos_stack = []

        super().__init__(env)
        image_height, image_width, channels = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(n_obs, image_height, image_width, channels),
            dtype=np.uint8,
        )

    def stack_observation(self, obs):
        # Append the new image to the stack
        if isinstance(obs, tuple):
            next_obs, reward, done, info = obs
            self.observations_stack.append(next_obs)
            self.rewards_stack.append(reward)
            self.dones_stack.append(done)
            self.infos_stack.append(info)

        # Append the new image to the stack
        elif isinstance(obs, np.ndarray):
            next_obs, reward, done, info = obs, 0, False, None
            self.observations_stack.append(next_obs)
            self.rewards_stack.append(reward)
            self.dones_stack.append(done)
            self.infos_stack.append(info)

        # Keep only the last n_obs Observation
        if len(self.observations_stack) > self.n_obs:
            self.observations_stack.pop(0)
            self.rewards_stack.pop(0)
            self.dones_stack.pop(0)
            self.infos_stack.pop(0)

    def reset(self):
        obs = super().reset()
        for _ in range(self.n_obs):
            self.stack_observation(obs)
        return (
            self.observations_stack,
            sum(self.rewards_stack) / self.n_obs,
            True if 1 == sum(self.dones_stack) else False,
            self.infos_stack,
        )

    def step(self, action):
        obs = super().step(action)
        self.stack_observation(obs)
        return (
            self.observations_stack,
            sum(self.rewards_stack) / self.n_obs,
            True if 1 == sum(self.dones_stack) else False,
            self.infos_stack,
        )


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_
