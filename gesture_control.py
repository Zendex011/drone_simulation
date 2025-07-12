import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data
import numpy as np
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found.")
    exit()

# Drone control
drone_pos, _ = p.getBasePositionAndOrientation(drone_id)
drone_pos = list(drone_pos)  # ✅ This is the fix

target_pos = list(drone_pos)
movement_speed = 0.1

# Helper functions
def is_fist_open(landmarks):
    """Return True if hand is open (fingers extended)"""
    return all(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20])

def is_fist_closed(landmarks):
    """Return True if hand is closed (fingers curled)"""
    return all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])

def thumb_direction(landmarks):
    """Return 'left' or 'right' based on thumb direction"""
    if landmarks[4].x < landmarks[3].x:
        return "left"
    elif landmarks[4].x > landmarks[3].x:
        return "right"
    return None

# Movement function
def move_drone(command):
    global drone_pos, target_pos
    x, y, z = drone_pos

    if command == "up":
        target_pos[2] += movement_speed
    elif command == "down":
        target_pos[2] -= movement_speed
    elif command == "tilt_left":
        target_pos[0] -= movement_speed
    elif command == "tilt_right":
        target_pos[0] += movement_speed

    # Smooth movement
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * 0.2
    print(f"Drone command: {command} | Position: {drone_pos}")
    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    command = "hover"

    if results.multi_hand_landmarks:
        hands_detected = results.multi_hand_landmarks

        # Single hand gesture
        if len(hands_detected) == 1:
            landmarks = hands_detected[0].landmark
            mp_drawing.draw_landmarks(frame, hands_detected[0], mp_hands.HAND_CONNECTIONS)

            if is_fist_open(landmarks):
                command = "up"
            elif is_fist_closed(landmarks):
                command = "down"

        # Two hand gesture (check for tilt)
        elif len(hands_detected) == 2:
            hand1 = hands_detected[0].landmark
            hand2 = hands_detected[1].landmark
            mp_drawing.draw_landmarks(frame, hands_detected[0], mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, hands_detected[1], mp_hands.HAND_CONNECTIONS)

            if is_fist_closed(hand1) and is_fist_closed(hand2):
                thumb1 = thumb_direction(hand1)
                thumb2 = thumb_direction(hand2)

                if thumb1 == "left" and thumb2 == "left":
                    command = "tilt_left"
                elif thumb1 == "right" and thumb2 == "right":
                    command = "tilt_right"

    move_drone(command)

    # UI
    cv2.putText(frame, f"Command: {command}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Drone Gesture Control", frame)

    p.stepSimulation()
    time.sleep(1 / 30)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
