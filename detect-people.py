# Beyblade detection
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
source = "People-Detection.mp4"
results = model(source, stream=False, save=True, conf=0.5, imgsz=640)