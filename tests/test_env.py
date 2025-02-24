from duckietown_rl_course.duckietownrl.gym_duckietown.envs import DuckietownEnv
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import time



parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--max_steps", default=1500, help="number of steps per episode", type=int)
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
parser.add_argument("--reward_invalid_pose", default=-100, type=int)
parser.add_argument("--alpha", default=0.20, type=float)
parser.add_argument("--collect_random_steps", default=3000, type=int)
parser.add_argument("--path", default="/media/g.ferraro/DONNEES", type=str)

args = parser.parse_args()

yellow_mask = True
n_frames = args.n_frames
n_chans = args.n_chans  # 1 for B/W images 3 for RGBs
n_envs = args.n_envs
resize_shape = (args.width_frame, args.height_frame)  # (width,height)

# initialize the environment
env = DuckietownEnv(
        map_name=args.map_name,
        distortion=args.distortion,
        domain_rand=args.domain_rand,
        max_steps=args.max_steps,
        seed=args.seed ,
        camera_width=resize_shape[0],
        camera_height=resize_shape[1],
        reward_invalid_pose=args.reward_invalid_pose,
        full_transparency=True,
    )

# env loop with rendering
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")
s = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    s, r, done, _ = env.step(action)
    print(f"s: {s.shape}, a: {action}, r: {r}, done: {done}")
    cv2.imshow("state", s)
    cv2.waitKey(1)
    # time.sleep(0.5)

