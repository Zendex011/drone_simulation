from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' is the nano version, fast and lightweight

print("YOLOv8 Model Loaded Successfully!")
