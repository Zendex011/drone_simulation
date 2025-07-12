import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
p.connect(p.GUI)

# Set search path for models
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load Ground Plane
plane_id = p.loadURDF("plane.urdf")

# Load Stadium
try:
    stadium_id = p.loadSDF("stadium.sdf")
    print("Stadium loaded successfully!")
except Exception as e:
    print(f"Error loading stadium: {e}")

# Load Drone Model
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])

# Set Gravity
p.setGravity(0, 0, -9.8)

# Adjust Camera Position
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 1])

# Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define Blue Color Range for Pen Detection
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Function to Map 2D Webcam Coordinates to 3D PyBullet Space
def map_2d_to_3d(x, y, width, height):
    x_3d = (x - width // 2) / 30  # Scale down movement
    y_3d = (height // 2 - y) / 30  # Flip y-axis
    z_3d = 1 + (height // 2 - y) / 100  # Adjust altitude dynamically
    return x_3d, y_3d, z_3d

# Smooth Movement Function
def move_drone_smoothly(target_pos, steps=10):
    current_pos, _ = p.getBasePositionAndOrientation(drone_id)
    for i in range(1, steps + 1):
        interp_pos = [
            current_pos[0] + (target_pos[0] - current_pos[0]) * i / steps,
            current_pos[1] + (target_pos[1] - current_pos[1]) * i / steps,
            current_pos[2] + (target_pos[2] - current_pos[2]) * i / steps,
        ]
        p.resetBasePositionAndOrientation(drone_id, interp_pos, [0, 0, 0, 1])
        p.stepSimulation()
        time.sleep(1 / 120)  # Higher update rate for smooth motion

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not read")
        break

    height, width, _ = frame.shape  # Get frame size

    # Convert to HSV and Mask Blue Pen
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 300:  # Minimum area to filter noise
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get Pen Center
            pen_x, pen_y = x + w // 2, y + h // 2

            # Convert to 3D Coordinates
            target_x, target_y, target_z = map_2d_to_3d(pen_x, pen_y, width, height)

            # Print debug information
            print(f"Moving Drone to: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")

            # Smoothly Move Drone in PyBullet
            move_drone_smoothly([target_x, target_y, target_z])

    # Show Camera Feed
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Mask", mask)

    # Step PyBullet Simulation
    p.stepSimulation()
    time.sleep(1 / 60)  # Run at 60 FPS

    # Exit Condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
p.disconnect()
