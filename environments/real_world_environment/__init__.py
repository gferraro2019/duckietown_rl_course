from gymnasium.envs.registration import register

register(
    id="DuckieBotDiscrete-v1",  # Unique name
    entry_point="environments.real_world_environment.duckie_bot_discrete:DuckieBotDiscrete",
    kwargs={
        "robot_name": "gastone",
        "stochasticity": 0.0
    }
)

from .duckie_bot_discrete import DuckieBotDiscrete

