from environments.real_world_environment import DuckieBotDiscrete

environment = DuckieBotDiscrete(robot_name="gastone")

print("[main] Environment initialised, sending action 0 on topic", environment.actions_publisher.name, "...")
environment.step(0)
print("[main] action executed.")