import pybullet as p
import pybullet_data
import time

# Connect to PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path for built-in assets
p.setGravity(0, 0, -9.8)  # Apply gravity

# Load the drone URDF (ensure the path is correct)
drone = p.loadURDF(r"C:\Users\ASUS\OneDrive\Desktop\college\sem 6\DRONES\project\cf2x.urdf", [0, 0, 1], useFixedBase=False)

# Keep the simulation running
while True:
    p.stepSimulation()
    time.sleep(0.01)  # Small delay to maintain smooth simulation
