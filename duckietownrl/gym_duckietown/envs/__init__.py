# coding=utf-8
from gymnasium.envs.registration import register
# noinspection PyUnresolvedReferences
from duckietownrl.gym_duckietown.wrappers import DiscreteWrapper
from duckietownrl.gym_duckietown.envs.duckietown_env import DuckietownEnv
from duckietownrl.gym_duckietown.envs.duckiebot_env import DuckiebotEnv
from duckietownrl.gym_duckietown.envs.multimap_env import MultiMapEnv
from duckietownrl.gym_duckietown.envs.duckietown_discrete_env import DuckietownDiscretEnv



register(
    id='Duckietown-v0',
    entry_point='duckietownrl.gym_duckietown.envs:DuckietownEnv',
    max_episode_steps=1500,
)

register(
    id='Duckiebot-v0',
    entry_point='duckietownrl.gym_duckietown.envs:DuckiebotEnv',
    max_episode_steps=1500,
)

register(
    id='MultiMap-v0',
    entry_point='duckietownrl.gym_duckietown.envs:MultiMapEnv',
    max_episode_steps=1500,
)

register(
    id='DuckietownDiscrete-v0',
    entry_point='duckietownrl.gym_duckietown.envs:DuckietownDiscretEnv',
    max_episode_steps=1500,
)

