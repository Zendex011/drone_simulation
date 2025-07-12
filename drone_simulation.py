import gym
import numpy as np
import pybullet as p
import pybullet_data
import time


import os
print("Current Working Directory:", os.getcwd())


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        # Start PyBullet GUI mode
        self.physics_client = p.connect(p.GUI)  # Opens the PyBullet window
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load built-in models
        p.setGravity(0, 0, -9.8)

        # Load a simple drone model (quadrotor or other built-in URDF)
        self.drone = p.loadURDF("cf2p.urdf", [0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        if self.drone < 0:
            print("Failed to load URDF!")

        
 

    def step(self, action):
        # Apply larger force to see the movement
        p.applyExternalForce(self.drone, -1, [action[0], action[1], action[2]], [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        return np.zeros(3), 0, False, {}
    

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 1], [0, 0, 0, 1])
        return np.zeros(3)

    def render(self, mode="human"):
        pass

# Run the simulation
env = DroneEnv()
while True:
    env.step([0, 0, 500])  # Moves the drone upwards with stronger force
    time.sleep(0.01)  # Small delay to observe
