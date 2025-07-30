# Robot Arm Inverse Kinematics Solver using Reinforcement Learning

This project uses OpenAI's Gym, Stable Baselines 3 and panda-gym (2.0.0) to train a reinforcement learning model to solve the problem of inverse kinematics at joints level for a robot arm, in this case a Franka Emika Panda.

Included are the scripts used and the experimental results obtained, but also the models and screenshots or videos of several types of episodes met.

The two fully trained models are:
 - ``models/ppo_posrotran04_800kit_150ep``
 - ``models/sac_posrotran04_800kit_150ep``

## Resources

 - ``models/`` - contains model trained during the study
 - ``panda_gym/`` - contains version 2.0.0 of panda-gym with several modifications done (mentioned in the study)
 - ``plot_results/`` - contains the results and the plot of the training process in the first case study, comparing results from PPO and SAC
 - ``results/`` - contains the results from the second case study, comparing DLS (the traditional method used in panda-gym/pybullet) and the hybrid method using SAC, including videos and screenshots
 - ``compareAll.py`` - runs 1000 seeded episodes for each method, outputting the results in the console output in .csv format
 - ``compareOne.py`` - runs 1000 seeded episodes for only one of the specified methods, similar to the previous script
 - ``exampleClient.py`` - opens the simulation in rendered mode, allowing manual control of the target position using the sliders in pybullet's debug menu (press ``G`` to open); ``spacebar`` can be held to prevent the robot from solving while changing the target position; the method used here by default is that of the smoothed hybrid
 - ``hybrid.py`` - visualize the behavior of simple, threshold-based hybrids
 - ``proportional.py`` - visualize the behavior of an approach based on DLS, solving for relative displacements in Cartesian coordinates
 - ``resumse_training_PPO.py`` - resume the training on an existing PPO model
 - ``resumse_training_SAC.py`` - resume the training on an existing SAC model
 - ``smoothHybrid.py`` - visualize the behavior of the hybrid method based on interpolating between the RL method and the traditional method
 - ``trainPPO.py`` - train a new PPO model, outputting the training progress in plot_results
 - ``trainSAC.py`` - train a new SAC model, outputting the training progress in plot_results
 - ``video.py`` - record video of an episode
 - ``visualizePPO.py`` - visualize the behavior of the given PPO model
 - ``visualizeSAC.py`` - visualize the behavior of the given SAC model 

## Quick start

### Install

To run the project locally, make sure to use anaconda (miniconda in this case) and to install the requirements

```bash
pip install -r requirements.txt
```

### Training

To train a model, the following parameters should be tweaked prior to starting:

Insert a maximum number of steps per episode in the PandaCustomReach environment from ``panda_gym/__init__.py`` as shown below:
```py
        register(
            id="PandaCustomReach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaCustomReachEnv",
            kwargs=kwargs,
            max_episode_steps=150,
        )
```

Perturbation term that randomizes the starting position of the arm, based on the seed, can be modified in ``panda_gym/envs/robots/panda.py``, within the ``set_joint_random`` method
```py
    def set_joint_random(self) -> None:
        """Set the robot to a random pose."""
        random_joint_values = self.neutral_joint_values + (self.np_random.random_sample(9) * 2 - np.ones(9)) * (np.pi * 0.4)
```
In this example, the perturbation term is equal to ``0.4``.

Then, either of ``trainPPO.py`` or ``trainSAC.py`` can be used to train a model, which will be outputted as ``model.zip``. Training can be continued on top of existing models using the ``resume_training_PPO.py`` or ``resume_training_SAC.py`` scripts, respectively.

The models can then be used in the rest of the scripts as described in the Resources chapter of this README.

### License

This repo includes a copy of panda_gym, which is released under MIT license.