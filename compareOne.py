import gym
import panda_gym
from stable_baselines3 import SAC
#from stable_baselines3 import PPO
from panda_gym.utils import distance
import numpy as np
import time
import math

env = gym.make("PandaCustomReachJointsDense-v2", render=False)
seeds = env.seed(26)
orig_env = env
robot = orig_env.robot
model = SAC.load("models/sac_posrotran04_800kit_150ep")
#model = PPO.load("models/ppo_posrotran04_800kit_150ep")
SWAP_THRESHOLD = 0.1
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

def interpolateActions(far_action, close_action, distance):
    action = []
    far_weight = 2.5 * distance - 0.5
    close_weight = 1.5 - 2.5 * distance
    for i in range(len(far_action)):
        value = far_action[i] * np.clip(far_weight, 0, 1) + close_action[i] * np.clip(close_weight, 0, 1)
        action.append(value)
    return action

def solveSmoothHybrid(robot, model, obs):
    current_pos, target_pos = obs["achieved_goal"], obs["desired_goal"]
    d = distance(current_pos, target_pos)
    action = None
    predictedAction, _states = model.predict(obs, deterministic=True)
    dp = (target_pos - current_pos)
    #dp = np.clip(dp, MIN_ACTION_EE, MAX_ACTION_EE)
    target_angles = robot.ee_displacement_to_target_arm_angles(dp)
    current_angles = robot.get_joint_angles()[:7]
    jdp = (target_angles - current_angles) * 40
    jdp = np.clip(jdp, MIN_ACTION_JOINTS, MAX_ACTION_JOINTS)
    action = interpolateActions(predictedAction, jdp, d)
    return action

print("Seed,DLS,Hybrid-0.1,Hybrid-0.2,Smooth Hybrid")

for i in range(0, 100):
    env.seed(i)
    obs = env.reset()
    for it in range(0, 300):
        action = solveSmoothHybrid(robot, model, obs)
        obs, reward, done, info = env.step(action)

    current_pos, target_pos = obs["achieved_goal"], obs["desired_goal"]
    d = distance(current_pos, target_pos)
    print(f"{i},{d < 0.2}")