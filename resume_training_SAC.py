import os
import gym
import panda_gym
from stable_baselines3 import SAC

from stable_baselines3.common.logger import configure

log_dir = "./plot_results/"
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "csv"])

env = gym.make("PandaCustomReachJoints-v2")
#env = FlattenObservation(env)
model = SAC.load("sac_posrotran04_600kit_150ep")
model.set_env(env)
model.set_logger(new_logger)
model.learn(total_timesteps=200000, reset_num_timesteps=False)
model.save("sac_posrotran04_800kit_150ep")