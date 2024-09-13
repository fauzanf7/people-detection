from ultralytics import YOLO

model = YOLO("best.pt")
source = "video.mp4"
results = model(source, stream=False, save=True, conf=0.5, imgsz=640)