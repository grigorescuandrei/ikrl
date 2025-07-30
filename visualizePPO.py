import gym
import panda_gym
from stable_baselines3 import PPO

env = gym.make("PandaCustomReachJointsDense-v2", render=True)

model = PPO.load("models/ppo_posrotran025_800kit_150ep")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    #print(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()