import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import mediapipe as mp
from object_detector import predict_object_from_frame  # Your custom model

# === PyBullet Setup ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
stadium_id = p.loadSDF("stadium.sdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])
p.setGravity(0, 0, -9.8)
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 1])

# === Camera Init ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Target Object ===
target_object = "pen"  # Hardcoding pen as the target for simplicity

# === MediaPipe Hands Init ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === Functions ===
def spawn_obstacles(num_obstacles=5):
    obstacles = []
    for _ in range(num_obstacles):
        x, y, z = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(1, 2.5)
        obstacle_id = p.loadURDF("cube.urdf", basePosition=[x, y, z])
        obstacles.append((obstacle_id, x, y, z))
    return obstacles

# === State Init ===
obstacles = spawn_obstacles()
drone_pos = [0, 0, 1]
movement_speed = 0.1
target_pos = list(drone_pos)

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    # === STEP 1: Object Detection ===
    center_crop = frame[height//2 - 100:height//2 + 100, width//2 - 100:width//2 + 100]
    detected_label = predict_object_from_frame(center_crop)
    detection_found = detected_label == target_object

    if detection_found:
        # Get pen's position relative to the frame center (simple method based on color/contour, etc.)
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which should be the pen
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Calculate center of the pen
            pen_center = (x + w // 2, y + h // 2)

            # === Draw Moving Bounding Box ===
            cv2.rectangle(center_crop, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Bounding box around pen
            cv2.circle(center_crop, pen_center, 5, (0, 0, 255), -1)  # Center of the pen as red dot

            # === Drone Follows Target (pen) ===
            # Adjust the drone's position based on the pen's location in the frame
            if pen_center[0] < width // 2 - 50:  # Pen is on the left side
                target_pos[0] -= 0.1
            elif pen_center[0] > width // 2 + 50:  # Pen is on the right side
                target_pos[0] += 0.1

            # If pen is far in vertical direction, move drone up or down
            if pen_center[1] < height // 2 - 50:  # Pen is above center
                target_pos[2] += 0.1
            elif pen_center[1] > height // 2 + 50:  # Pen is below center
                target_pos[2] -= 0.1

    # === Smooth Drone Movement ===
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * movement_speed
    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])

    # === Display Info ===
    cv2.putText(frame, f"Tracking: {detected_label}" if detection_found else "No target detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Show Camera Feed with Pen Detection ===
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Detection Zone", center_crop)  # Show crop of the detected pen

    p.stepSimulation()
    time.sleep(1 / 60)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
p.disconnect()
