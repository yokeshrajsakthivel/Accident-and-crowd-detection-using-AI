import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model (Download if not found)
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    from ultralytics import YOLO
    YOLO(model_path)  # Downloads the model if missing

# Load the YOLOv8 model
model = YOLO(model_path)  

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def detect_people(frame):
    # Run YOLOv8 object detection
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0].item())  # Convert confidence to float
            cls = int(box.cls[0].item())  # Class ID

            # Check if detected class is 'person' (COCO class ID 0)
            if cls == 0 and conf > 0.5:
                detections.append(([x1, y1, x2, y2], conf, cls))

    # Track detected people using DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    people_detected = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        people_detected.append((x1, y1, x2, y2))

    return people_detected