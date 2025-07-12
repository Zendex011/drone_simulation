# Drone Simulation Project

This project simulates a drone in a 3D environment using PyBullet and allows control via object detection (pen, pencil, mechanical pen) and hand gestures. The simulation uses a webcam for real-time input and a custom-trained CNN model for object recognition.

## Demo
You can watch a demo video of the project here:
- [Demo Video 1](https://drive.google.com/file/d/1jwWqG-SREtZ30eZMcciQe1JClYtk2iJe/view?usp=drivesdk)
- [Demo Video 2](https://drive.google.com/file/d/1jwWqG-SREtZ30eZMcciQe1JClYtk2iJe/view?usp=drivesdk)

## Main Features
- **Object Tracking:** The drone follows a specified object (e.g., pen) detected in the webcam feed.
- **Gesture Control:** Control the drone using hand gestures (requires gesture_control module).
- **Obstacle Avoidance:** Random obstacles are spawned in the simulation environment.
- **Multiple Modes:** Choose between gesture control, pen detection, or hover mode (see `test123.py`).

## Usage
1. Clone this repository:
   ```sh
   git clone https://github.com/Zendex011/drone_simulation.git
   cd drone_simulation
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the main simulation:
   ```sh
   python main.py
   ```
   Or run the multi-mode script:
   ```sh
   python test123.py
   ```

## Note on Model Files
**The file `yolov3.weights` is required for some object detection features but is not included in this repository due to GitHub's file size limits.**

- You must manually download `yolov3.weights` and place it in the project directory.
- You can download it from the official YOLO website: https://pjreddie.com/darknet/yolo/

## Datasets
- The datasets used for training and validation are not included in this repository to keep the repo size manageable.

## File Overview
- `main.py`: Runs the basic drone simulation with object tracking.
- `test123.py`: Allows selection of control mode (gesture, pen detection, hover).
- `object_detector.py`: Loads the CNN model and provides object prediction from webcam frames.
- `gesture_control.py`: (If implemented) Provides gesture recognition from webcam frames.

## License
This project is for educational and research purposes. 