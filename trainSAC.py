import os
import gym
import panda_gym
from stable_baselines3 import SAC

from stable_baselines3.common.logger import configure

log_dir = "./plot_results/"
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "csv"])

env = gym.make("PandaCustomReachJoints-v2")
model = SAC("MultiInputPolicy", env, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=200000)
model.save("model")