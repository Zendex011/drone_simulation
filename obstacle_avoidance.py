import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load Ground Plane
plane_id = p.loadURDF("plane.urdf")

# Load Stadium
stadium_id = p.loadSDF("stadium.sdf")

# Load Drone Model
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])

# Set Gravity
p.setGravity(0, 0, -9.8)

# Function to Perform Raycasting for Obstacle Detection
def detect_obstacles(drone_pos):
    directions = [
        (1, 0, 0), (-1, 0, 0),  # Forward, Backward
        (0, 1, 0), (0, -1, 0),  # Left, Right
        (0, 0, 1), (0, 0, -1)   # Up, Down
    ]
    obstacle_detected = False
    safe_direction = None

    for direction in directions:
        ray_end = np.add(drone_pos, np.multiply(direction, 2))  # Extend 2m in each direction
        ray_test = p.rayTest(drone_pos, ray_end)
        hit_object = ray_test[0][0]

        if hit_object >= 0:
            obstacle_detected = True
        else:
            safe_direction = direction  # Store a safe direction

    return obstacle_detected, safe_direction

# Main Simulation Loop
while True:
    drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
    obstacle, safe_dir = detect_obstacles(drone_pos)
    
    if obstacle and safe_dir:
        # Move drone in a safe direction to avoid collision
        new_pos = np.add(drone_pos, np.multiply(safe_dir, 0.1))
        p.resetBasePositionAndOrientation(drone_id, new_pos, [0, 0, 0, 1])
    
    p.stepSimulation()
    time.sleep(1/60)  # 60 FPS
    
    if p.getKeyboardEvents().get(ord('q')):
        break

# Cleanup
p.disconnect()
