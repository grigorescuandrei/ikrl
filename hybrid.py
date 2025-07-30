import gym
import panda_gym
from stable_baselines3 import SAC
#from stable_baselines3 import PPO
from panda_gym.utils import distance
import numpy as np
import time
import math

env = gym.make("PandaCustomReachJointsDense-v2", render=True)
seeds = env.seed(430)
print("seeds:", seeds)
orig_env = env
robot = orig_env.robot
model = SAC.load("models/sac_posrotran04_800kit_150ep")
#model = PPO.load("models/ppo_posrotran025_800kit_150ep")
SWAP_THRESHOLD = 100.0
MIN_ACTION_EE = -np.ones(3)
MAX_ACTION_EE = np.ones(3)
MIN_ACTION_JOINTS = -np.ones(7)
MAX_ACTION_JOINTS = np.ones(7)

def solveHybrid(robot, model, obs):
    current_pos, target_pos = obs["achieved_goal"], obs["desired_goal"]
    d = distance(current_pos, target_pos)
    action = None
    if d > SWAP_THRESHOLD:
      action, _states = model.predict(obs, deterministic=True)
    else:
      dp = (target_pos - current_pos)
      #dp = np.clip(dp, MIN_ACTION_EE, MAX_ACTION_EE)
      target_angles = robot.ee_displacement_to_target_arm_angles(dp)
      current_angles = robot.get_joint_angles()[:7]
      jdp = (target_angles - current_angles) * 40
      jdp = np.clip(jdp, MIN_ACTION_JOINTS, MAX_ACTION_JOINTS)
      action = jdp
    return action

obs = env.reset()
while True:
    action = solveHybrid(robot, model, obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()