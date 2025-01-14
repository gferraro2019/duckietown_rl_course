from duckietown_rl_course.duckietownrl.gym_duckietown import envs 
import gymnasium as gym
import cv2
env = gym.make('DuckietownDiscrete-v0')

print(env.action_space)
print(env.observation_space)
obs, _ = env.reset()
print(obs.shape)
for _ in range(1000):
    action = env.action_space.sample()
    obs, _, _, _,info = env.step(action)
    print(obs.shape)
    cv2.imshow('obs', obs[:,:,0:3])
    cv2.waitKey(10)