import gym
import panda_gym

env = gym.make("PandaCustomReachDense-v2", render=True)

def chooseAction(currentPos, targetPos):
    """
    compute action to take proportionally to error
    """
    n = len(currentPos)
    p = 10
    
    action = p * (targetPos - currentPos)

    return action


env.reset()
action = env.action_space.sample()
while True:
    obs, reward, done, info = env.step(action)
    currentPos = obs["achieved_goal"]
    targetPos = obs["desired_goal"]
    action = chooseAction(currentPos, targetPos)
    env.render()
    if done:
      obs = env.reset()