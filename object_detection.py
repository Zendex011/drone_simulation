import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import mediapipe as mp
from ultralytics import YOLO

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load Environment
plane_id = p.loadURDF("plane.urdf")
stadium_id = p.loadSDF("stadium.sdf")

drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])
p.setGravity(0, 0, -9.8)

# Adjust Camera
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 1])

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Get user input for object detection
target_object = input("Enter the object to detect: ").strip().lower()

# MediaPipe Hands for Gesture Control
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def map_2d_to_3d(x, y, width, height):
    return (x - width // 2) / 30, (height // 2 - y) / 30, 1.5

def spawn_obstacles(num_obstacles=5):
    obstacles = []
    for _ in range(num_obstacles):
        x, y, z = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(1, 2.5)
        obstacle_id = p.loadURDF("cube_small.urdf", basePosition=[x, y, z])
        obstacles.append((obstacle_id, x, y, z))
    return obstacles

def check_collision(drone_pos):
    x, y, z = drone_pos
    return any(abs(x - ox) < 0.5 and abs(y - oy) < 0.5 and abs(z - oz) < 0.5 for _, ox, oy, oz in obstacles)

def detect_gesture(landmarks):
    thumb_tip, thumb_base = landmarks[4].y, landmarks[2].y
    index_tip, middle_tip, ring_tip, pinky_tip = landmarks[8].y, landmarks[12].y, landmarks[16].y, landmarks[20].y
    if thumb_tip < thumb_base: return "ascend"
    if thumb_tip > thumb_base: return "descend"
    if index_tip < middle_tip and index_tip < ring_tip and index_tip < pinky_tip: return "forward"
    if index_tip > middle_tip and index_tip > ring_tip and index_tip > pinky_tip: return "backward"
    if landmarks[5].x < landmarks[17].x: return "left"
    if landmarks[5].x > landmarks[17].x: return "right"
    return "hover"

obstacles = spawn_obstacles()
drone_pos = [0, 0, 1]
movement_speed = 0.05

while True:
    ret, frame = cap.read()
    if not ret: break
    height, width, _ = frame.shape

    # Run YOLOv8 object detection
    results = model(frame)
    detection_found = False
    
    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls)]  # Get class name
            if cls.lower() == target_object:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detection_found = True
    
    command = "hover"
    target_pos = list(drone_pos)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            command = detect_gesture(hand_landmarks.landmark)
    
    if command == "ascend": target_pos[2] += 0.3
    elif command == "descend": target_pos[2] -= 0.3
    elif command == "left": target_pos[0] -= 0.3
    elif command == "right": target_pos[0] += 0.3
    elif command == "forward": target_pos[1] += 0.3
    elif command == "backward": target_pos[1] -= 0.3
    
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * movement_speed
    
    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])
    
    cv2.putText(frame, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if detection_found:
        cv2.putText(frame, "Object Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Camera Feed", frame)
    p.stepSimulation()
    time.sleep(1 / 60)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
