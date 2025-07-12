'''
import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import mediapipe as mp
from object_detector import predict_object_from_frame  # <- Your custom model

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load Environment
plane_id = p.loadURDF("plane.urdf")
stadium_id = p.loadSDF("stadium.sdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])
p.setGravity(0, 0, -9.8)
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 1])

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get user input for object detection
target_object = input("Enter the object to detect (pen/pencil/mechanical_pen): ").strip().lower()

# MediaPipe Hands for Gesture Control
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to map 2D image coordinates to 3D space
def map_2d_to_3d(x, y, width, height):
    return (x - width // 2) / 30, (height // 2 - y) / 30, 1.5

# Spawn random obstacles
def spawn_obstacles(num_obstacles=5):
    obstacles = []
    for _ in range(num_obstacles):
        x, y, z = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(1, 2.5)
        obstacle_id = p.loadURDF("cube_small.urdf", basePosition=[x, y, z])
        obstacles.append((obstacle_id, x, y, z))
    return obstacles

# Detect collisions
def check_collision(drone_pos):
    x, y, z = drone_pos
    return any(abs(x - ox) < 0.5 and abs(y - oy) < 0.5 and abs(z - oz) < 0.5 for _, ox, oy, oz in obstacles)

# Gesture classification
def detect_gesture(landmarks):
    # Calculate distances between fingertips and base joints for gesture classification
    def is_finger_extended(tip, base): return tip.y < base.y  # y decreases upward in image space

    fingers = {
        "thumb": is_finger_extended(landmarks[4], landmarks[2]),
        "index": is_finger_extended(landmarks[8], landmarks[6]),
        "middle": is_finger_extended(landmarks[12], landmarks[10]),
        "ring": is_finger_extended(landmarks[16], landmarks[14]),
        "pinky": is_finger_extended(landmarks[20], landmarks[18]),
    }

    if fingers["thumb"] and not any([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]):
        return "ascend"
    elif not fingers["thumb"] and all([not fingers[f] for f in ["index", "middle", "ring", "pinky"]]):
        return "descend"
    elif fingers["index"] and not fingers["middle"]:
        return "forward"
    elif fingers["middle"] and not fingers["index"]:
        return "backward"
    elif landmarks[5].x < landmarks[17].x - 0.1:  # Left gesture
        return "left"
    elif landmarks[5].x > landmarks[17].x + 0.1:  # Right gesture
        return "right"
    else:
        return "hover"


# Initialize drone
obstacles = spawn_obstacles()
drone_pos = [0, 0, 1]
movement_speed = 0.1

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    detection_found = False

    # ---- STEP 1: Predict object using your CNN model ----
    predicted_object = predict_object_from_frame(frame)

    if predicted_object == target_object:
        detection_found = True
        cv2.putText(frame, f"Detected: {predicted_object}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---- STEP 2: Gesture Detection ----
    command = "hover"
    target_pos = list(drone_pos)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        print(f"[DEBUG] Hand landmarks detected: {len(hand_results.multi_hand_landmarks)}")

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            command = detect_gesture(hand_landmarks.landmark)

    # ---- STEP 3: Gesture-based Movement ----
    if command == "ascend":
        target_pos[2] += 0.3
    elif command == "descend":
        target_pos[2] -= 0.3
    elif command == "left":
        target_pos[0] -= 0.3
    elif command == "right":
        target_pos[0] += 0.3
    elif command == "forward":
        target_pos[1] += 0.3
    elif command == "backward":
        target_pos[1] -= 0.3

    # Smooth transition
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * movement_speed

    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])

    # ---- STEP 4: Display Status ----
    cv2.putText(frame, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if detection_found:
        cv2.putText(frame, "🎯 Target Object Found!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Camera Feed", frame)

    p.stepSimulation()
    time.sleep(1 / 60)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()
p.disconnect()
'''

# test123.py

import cv2
import pybullet as p
import pybullet_data
import time
import sys

# ====== Optional Import ======
from object_detector import predict_object_from_frame  # Used in mode 2 (pen detection)
from gesture_control import get_gesture_prediction    # Create this if not yet done

# ==== Control Mode Selection ====
print("Select Control Mode:")
print("1: Gesture Control")
print("2: Pen Detection")
print("3: Hover")
control_mode = int(input("Enter mode (1/2/3): "))

# ==== PyBullet Setup ====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])
drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
drone_pos = list(drone_pos)
target_pos = list(drone_pos)
movement_speed = 0.1

def move_drone(command):
    global drone_pos, target_pos

    # Force convert to list in case they're still tuples
    drone_pos = list(drone_pos)
    target_pos = list(target_pos)

    if command == "up":
        target_pos[2] += movement_speed
    elif command == "down":
        target_pos[2] -= movement_speed
    elif command == "left":
        target_pos[0] -= movement_speed
    elif command == "right":
        target_pos[0] += movement_speed

    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * 0.2

    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])



def move_drone_based_on_tile_position(row, col, grid_size):
    center_row, center_col = grid_size[0] // 2, grid_size[1] // 2
    if row < center_row:
        target_pos[1] += movement_speed  # Forward
    elif row > center_row:
        target_pos[1] -= movement_speed  # Backward
    if col < center_col:
        target_pos[0] -= movement_speed  # Left
    elif col > center_col:
        target_pos[0] += movement_speed  # Right
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * 0.2
    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])

# ==== Webcam Setup ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    sys.exit()

# ==== Main Loop ====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Mode 1: Gesture Control ===
    if control_mode == 1:
        gesture = get_gesture_prediction(frame)  # Define this function
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

        if gesture == "open_fist":
            move_drone(dz=movement_speed)  # Up
        elif gesture == "closed_fist":
            move_drone(dz=-movement_speed)  # Down
        elif gesture == "left_thumb":
            move_drone(dx=-movement_speed)  # Left
        elif gesture == "right_thumb":
            move_drone(dx=movement_speed)  # Right

    # === Mode 2: Pen Detection ===
    elif control_mode == 2:
        label = predict_object_from_frame(tile)
    

        height, width, _ = frame.shape
        tile_size = 128
        rows = height // tile_size
        cols = width // tile_size
        detected = False

        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * tile_size, (r + 1) * tile_size
                x1, x2 = c * tile_size, (c + 1) * tile_size
                tile = frame[y1:y2, x1:x2]

                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    continue

                label = predict_object_from_frame(tile)
                if label == "pen":
                    detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    move_drone_based_on_tile_position(r, c, (rows, cols))
                    break
            if detected:
                break

    # === Mode 3: Hover ===
    elif control_mode == 3:
        cv2.putText(frame, "Hover Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # === Common Tasks ===
    cv2.imshow("Drone View", frame)
    p.stepSimulation()
    time.sleep(1 / 30)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
