import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time

# Initialize PyBullet simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load a drone model (replace with an appropriate URDF file)
planeId = p.loadURDF("plane.urdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])
if drone_id < 0:
    print("Failed to load drone model!")


# Start video capture
cap = cv2.VideoCapture(0)

# Define color range for detecting a pen (adjust as needed)
lower_color = np.array([30, 150, 50])
upper_color = np.array([80, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours of the detected object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the pen)
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # X coordinate
            cy = int(M["m01"] / M["m00"])  # Y coordinate

            # Draw marker on the detected pen position
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Map 2D (cx, cy) to 3D PyBullet (x, y)
            x_pos = (cx / frame.shape[1]) * 2 - 1  # Normalize to [-1, 1]
            y_pos = (cy / frame.shape[0]) * 2 - 1

            # Move the drone
            p.resetBasePositionAndOrientation(droneId, [x_pos, y_pos, 1], [0, 0, 0, 1])

    # Show the frame with tracking
    cv2.imshow("Pen Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
p.disconnect()
