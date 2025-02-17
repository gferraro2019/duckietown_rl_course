import gymnasium as gym
import pandas as pd
from pynput import keyboard
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime

# Import the DuckieBot environment
from environments.real_world_environment import RealWorldEnvironment

env = gym.make('DuckieBotDiscrete-v1', render_mode="human")
obs, info = env.reset()

data = []  # To store interaction samples

key_action_map = {
    keyboard.Key.up: 0,  # Forward
    keyboard.Key.down: 1,  # Backward
    keyboard.Key.left: 2,  # Turn left
    keyboard.Key.right: 3  # Turn right
}

current_action = 4  # Default to no action


def on_press(key):
    global current_action
    if key in key_action_map:
        current_action = key_action_map[key]
    elif key == keyboard.Key.esc:
        return False  # Stop listener


def on_release(key):
    global current_action
    if key in key_action_map:
        current_action = 4  # Reset to no action


print("Use arrow keys to control the DuckieBot. Press 'Esc' to quit.")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while True:
        if current_action == 4:
            continue
        next_obs, reward, terminated, truncated, info = env.step(current_action)

        # Store interaction sample
        data.append({
            "action": current_action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": str(info)
        })

        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()
    listener.stop()

    # Save to Parquet file
    if data:
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, "duckiebot_interactions.parquet")
        print("Dataset saved as duckiebot_interactions.parquet")
    else:
        print("No interactions recorded.")