from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        print("⏳ Loading YOLO model...")
        self.model = YOLO("./yolov8n.pt")   # ✅ FIXED PATH
        print("✅ YOLO loaded")

    def detect(self, frame):
        return self.model(frame)