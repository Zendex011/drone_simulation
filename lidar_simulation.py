import pybullet as p
import pybullet_data
import numpy as np
import time
import open3d as o3d  # For 3D visualization

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load Ground Plane & Drone
plane_id = p.loadURDF("plane.urdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])

# LiDAR Parameters
NUM_RAYS = 360  # 360-degree scan
MAX_DISTANCE = 10  # Max LiDAR range

def perform_lidar_scan():
    """Simulate a LiDAR scan by casting rays in a 360-degree pattern."""
    drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
    start_pos = np.array(drone_pos)
    lidar_hits = []
    
    for angle in np.linspace(0, 2 * np.pi, NUM_RAYS):
        direction = np.array([np.cos(angle), np.sin(angle), 0])
        end_pos = start_pos + direction * MAX_DISTANCE
        ray_id = p.rayTest(start_pos.tolist(), end_pos.tolist())[0]
        hit_position = ray_id[3]
        
        if ray_id[0] != -1:  # Valid hit
            lidar_hits.append(hit_position)
    
    return np.array(lidar_hits)

def visualize_lidar_data(point_cloud):
    """Display the LiDAR data as a 3D point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

# Main Simulation Loop
while True:
    lidar_data = perform_lidar_scan()
    if len(lidar_data) > 0:
        print(f"Captured {len(lidar_data)} LiDAR points")
    
    p.stepSimulation()
    time.sleep(1 / 10)  # Run at 10 Hz
    
    # Exit condition
    if input("Press 'q' to quit, Enter to continue: ") == 'q':
        break

p.disconnect()
