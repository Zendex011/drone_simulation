import cv2
import numpy as np
import threading

# Load YOLO model
yolo_config = "yolov3.cfg"
yolo_weights = "yolov3.weights"
yolo_classes = "coco.names"

# Read class labels
with open(yolo_classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the YOLOv3 model
net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Shared variable to store the latest detected object position
detected_object_position = None

def detect_object():
    global detected_object_position
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if frame is not captured

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    detected_object_position = (center_x, center_y)  # Store latest position

                    # Draw bounding box
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2),
                                  (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                    cv2.putText(frame, classes[class_id], (center_x - 10, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    break  # Only track the first detected object

        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()

# Start YOLO detection in a separate thread
yolo_thread = threading.Thread(target=detect_object, daemon=True)
yolo_thread.start()
