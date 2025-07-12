import cv2
import numpy as np

# Load YOLO model
yolo_config = "yolov3.cfg"  # Ensure this file is in the same folder as your script
yolo_weights = "yolov3.weights"
yolo_classes = "coco.names"

# Read class labels
with open(yolo_classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the YOLOv3 model
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

print("YOLO model loaded successfully!")

