'''
# drone_controller.py

import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data
import numpy as np
import time
from object_detector import predict_object_from_frame

# Init Bullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")
drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])

drone_pos = list(p.getBasePositionAndOrientation(drone_id)[0])
target_pos = list(drone_pos)
movement_speed = 0.1

# Init Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible.")
    p.disconnect()
    exit()

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drone Movement
def move_drone(dx=0, dy=0, dz=0):
    global drone_pos, target_pos
    target_pos[0] += dx
    target_pos[1] += dy
    target_pos[2] += dz
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * 0.2
    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])

# Gesture Logic (updated and reliable)
def detect_gesture(landmarks):
    lm = landmarks.landmark

    # Check each finger tip against MCP (Metacarpophalangeal joint)
    fingers_up = [
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_MCP].y,
    ]

    if all(fingers_up):
        return "fist_open"
    elif not any(fingers_up):
        return "fist_closed"
    else:
        return "none"

# ==========================
# MODE SELECTION
# ==========================
print("\n🌐 Select Control Mode:")
print("1. Gesture Control (✊ / 🖐️)")
print("2. Pen Detection (Pen moves drone forward)")
print("3. Autonomous Hover")
choice = input("Enter your choice (1/2/3): ").strip()

if choice not in ['1', '2', '3']:
    print("❌ Invalid choice. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()
    exit()

print(f"✅ Mode {choice} selected. Starting drone control...")

# ==========================
# MAIN LOOP
# ==========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        if choice == '1':  # Gesture Control
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture = detect_gesture(hand_landmarks)
                    print("👉 Detected gesture:", gesture)  # Debug print

                    if gesture == "fist_open":
                        move_drone(dz=movement_speed)
                        cv2.putText(frame, "Gesture: UP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif gesture == "fist_closed":
                        move_drone(dz=-movement_speed)
                        cv2.putText(frame, "Gesture: DOWN", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Gesture: None", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            else:
                cv2.putText(frame, "No Hand Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, "Mode: Gesture Control", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif choice == '2':  # Pen Detection
            center_crop = frame[height // 3:2 * height // 3, width // 3:2 * width // 3]
            prediction = predict_object_from_frame(center_crop)
            cv2.putText(frame, f"Detected: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if prediction == "pen":
                move_drone(dy=movement_speed)
                cv2.putText(frame, "Pen Detected: Moving Forward", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif choice == '3':  # Autonomous Hover
            cv2.putText(frame, "Mode: Hovering", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

        cv2.imshow("Drone View", frame)
        p.stepSimulation()
        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting simulation...")
            break

except Exception as e:
    print("❌ Runtime Error:", e)

# Cleanup
cap.release()
cv2.destroyAllWindows()
p.disconnect() '''
import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data
import numpy as np
import time
from object_detector import predict_object_from_frame
import tkinter as tk
from tkinter import simpledialog, messagebox


def move_drone(drone_id, drone_pos, target_pos, dx=0, dy=0, dz=0):
    movement_speed = 0.1
    target_pos[0] += dx
    target_pos[1] += dy
    target_pos[2] += dz
    for i in range(3):
        drone_pos[i] += (target_pos[i] - drone_pos[i]) * 0.2
    p.resetBasePositionAndOrientation(drone_id, drone_pos, [0, 0, 0, 1])
    return drone_pos, target_pos


def ask_mode():
    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.update()
        choice = simpledialog.askstring(
            "Drone Mode Selection",
            "Enter mode:\n1 = Gesture Control\n2 = Pen Detection\n3 = Autonomous Hover",
            parent=root
        )
        root.destroy()
        return choice
    except Exception as e:
        print("⚠️ GUI input failed. Falling back to CLI input.")
        return input("Enter mode (1 = Gesture, 2 = Pen, 3 = Hover): ")


def main():
    # ======================= MODE SELECTION =======================
    print("======== Drone Mode Selection ========")
    print("1 = Gesture Control")
    print("2 = Pen Detection")
    print("3 = Autonomous Hover")
    choice = input("Enter mode number (1/2/3): ").strip()

    if not choice or choice not in ['1', '2', '3']:
        print("❌ Invalid or no input. Exiting...")
        return


    # ========== PYBULLET SETUP ==========
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    plane_id = p.loadURDF("plane.urdf")
    drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])

    # ========== CAMERA SETUP ==========
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible.")
        p.disconnect()
        return

    # ========== DRONE SETUP ==========
    drone_pos = list(p.getBasePositionAndOrientation(drone_id)[0])
    target_pos = list(drone_pos)
    movement_speed = 0.1

    try:
        if choice == '1':
            print("✅ Gesture mode started.")
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        lm = hand_landmarks.landmark
                        fingers = [
                            lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                            lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                            lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                            lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y
                        ]
                        if all(fingers):
                            drone_pos, target_pos = move_drone(drone_id, drone_pos, target_pos, dz=movement_speed)
                            cv2.putText(frame, "Gesture: UP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        elif not any(fingers):
                            drone_pos, target_pos = move_drone(drone_id, drone_pos, target_pos, dz=-movement_speed)
                            cv2.putText(frame, "Gesture: DOWN", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                cv2.putText(frame, "Mode: Gesture Control", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                cv2.imshow("Drone View", frame)
                p.stepSimulation()
                time.sleep(1/30)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif choice == '2':
            print("✅ Pen Detection mode started.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                height, width, _ = frame.shape
                center_crop = frame[height//3:2*height//3, width//3:2*width//3]
                prediction = predict_object_from_frame(center_crop)

                if prediction == "pen":
                    drone_pos, target_pos = move_drone(drone_id, drone_pos, target_pos, dy=movement_speed)
                    cv2.putText(frame, "Pen Detected: Moving Forward", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(frame, f"Detected: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow("Drone View", frame)
                p.stepSimulation()
                time.sleep(1/30)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif choice == '3':
            print("✅ Autonomous Hover started.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, "Mode: Hovering", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
                cv2.imshow("Drone View", frame)
                p.stepSimulation()
                time.sleep(1/30)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print("❌ Runtime Error:", e)

    # ========== CLEANUP ==========
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()


if __name__ == "__main__":
    main()
