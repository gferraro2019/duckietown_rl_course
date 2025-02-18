from gymnasium.envs.registration import register

register(
    id="DuckieBotDiscrete-v1",  # Unique name
    entry_point="environments.real_world_environment.duckie_bot_discrete:DuckieBotDiscrete",
    kwargs={
        "robot_name": "paperino",
        "fixed_linear_velocity": 0.3,
        "fixed_angular_velocity": 0.1,
        "action_duration": 0.3,
        "stochasticity": 0.1
    }
)
