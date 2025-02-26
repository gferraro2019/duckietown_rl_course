import pathlib

import gymnasium as gym
import pandas as pd
from pynput import keyboard
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime

# Import the DuckieBot environment
from environments.real_world_environment import DuckieBotDiscrete

env = DuckieBotDiscrete(robot_name="gastone")
obs, info = env.reset()

data = []  # To store interaction samples


key_action_map = {
    keyboard.Key.up: 0,         # Forward
    keyboard.Key.down: 1,       # Backward
    keyboard.Key.left: 2,       # Turn left
    keyboard.Key.right: 3,      # Turn right
    keyboard.Key.space: 4,      # Stop (4
    keyboard.Key.esc: 4         # Stop
}

current_action = None  # Default to no action


def on_press(key):
    global current_action
    if key in key_action_map:
        current_action = key_action_map[key]

def on_release(key):
    global current_action
    if key in key_action_map:
        current_action = None  # Reset to no action


print("Use arrow keys to control the DuckieBot. Press 'Space' or Esc to quit.")
robot_is_moving = False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

last_obs = None
try:
    while True:
        if current_action is None:
            if robot_is_moving:
                robot_is_moving = False
            continue
        if current_action == 4:
            print("current action is 4 STOOOP")
            break
        else:
            print("current action is ", current_action)
            robot_is_moving = True

        next_obs, reward, terminated, truncated, info = env.step(current_action)

        if last_obs is not None:
            print("data build")
            # Store interaction sample
            data.append({
                "s": last_obs.flatten().tolist(),
                "a": current_action,
                "r": reward,
                "d": terminated or truncated,
                "next_s": next_obs.flatten().tolist()
            })
        last_obs = next_obs.copy()

        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    print("keyboard interupted")
    pass
finally:
    print("In finally")
    env.close()
    listener.stop()

    # Save to Parquet file
    if data:
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)

        timestamp = datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")   # Get current date and time in the desired format
        current_dir = pathlib.Path(__file__).parent.absolute()
        filename = f"{current_dir}/duckiebot_interactions_{timestamp}.parquet"    # Convert the date-time into a filename
        df = pd.DataFrame(data)
        df.to_parquet(filename, engine="pyarrow", index=False)
    else:
        print("No interactions recorded.")