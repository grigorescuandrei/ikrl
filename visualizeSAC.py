import gym
import panda_gym
from stable_baselines3 import SAC
import random

env = gym.make("PandaCustomReachJointsDense-v2", render=True)
env.seed(1)
env.action_space.seed(1)

model = SAC.load("models/sac_posrotran04_800kit_150ep")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    #print(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()