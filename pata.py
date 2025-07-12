import cv2
import numpy as np
import threading
import time
import gym
import pybullet as p
import pybullet_envs
from stable_baselines3 import PPO

# ====================== YOLO Object Detection ====================== #
yolo_config = "yolov3.cfg"
yolo_weights = "yolov3.weights"
yolo_classes = "coco.names"

with open(yolo_classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

detected_object_position = None  # Shared variable

def detect_object():
    global detected_object_position
    cap = cv2.VideoCapture(0)
    print(f"Detected Object Position: {detected_object_position}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    detected_object_position = (center_x, center_y)
                    break

        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start YOLO in a separate thread
yolo_thread = threading.Thread(target=detect_object, daemon=True)
yolo_thread.start()

# ====================== Drone Environment (Gym) ====================== #
class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        self.drone = p.loadURDF("quadrotor.urdf", [0, 0, 1])

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)  # Fixed shape
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering temporarily
        self.drone = p.loadURDF("quadrotor.urdf", [0, 0, 1])   # Load the drone at (0,0,1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Enable rendering again


    def step(self, action):
        global detected_object_position  

        # Apply force to move drone
        p.applyExternalForce(self.drone, -1, [action[0], action[1], action[2]], [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        time.sleep(0.01)

        # Get drone position
        pos, _ = p.getBasePositionAndOrientation(self.drone)

        # Calculate reward based on distance to detected object
        reward = -1  # Default negative reward
        done = False

        if detected_object_position:
            target_x, target_y = detected_object_position
            drone_x, drone_y, _ = pos
            distance = np.linalg.norm(np.array([drone_x, drone_y]) - np.array([target_x, target_y]))

            reward = -distance  # Negative reward for farther distance

            if distance < 0.5:  # Close enough to the target
                reward = 100
                done = True  # Stop episode

        print(f"Drone Position: {pos}")

        return np.array(pos, dtype=np.float32), reward, done, {}

    def reset(self):
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 1], [0, 0, 0, 1])
        return np.array([0, 0, 1], dtype=np.float32)  # Fixed observation shape

# ====================== Train & Run the Model ====================== #
env = DroneEnv()
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Run the trained model
obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Use random actions
    obs, rewards, done, _ = env.step(action)
    time.sleep(0.1)
    
while True:  # Run indefinitely
    action, _ = model.predict(obs)
    obs, rewards, done, _ = env.step(action)
    
    if done:  # Reset only if needed
        obs = env.reset()

    time.sleep(0.05)  # Ensure stable execution speed
