import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.custom_reach import CustomReach
from panda_gym.pybullet import PyBullet


class PandaCustomReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        self.render_mode = "human"
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomReach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, get_ee_rotation=robot.get_ee_rotation, get_ee_velocity=robot.get_ee_velocity)
        super().__init__(robot, task)

    def set_goal(self, new_goal):
        self.task.set_goal(new_goal)
