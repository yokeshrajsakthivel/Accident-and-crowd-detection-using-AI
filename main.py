import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.ensemble import IsolationForest
from utils import draw_boxes, preprocess_features  # Import helper functions

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)

# Initialize Anomaly Detection Model (Isolation Forest)
anomaly_detector = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# Load Video Dataset (Change filename as needed)
video_path = "dataset/sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# Store detected people counts for anomaly detection
people_counts = []

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Detect people using YOLO
    detections = model(frame)[0]
    people = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == 0 and score > 0.5:  # Class 0 = Person
            people.append([x1, y1, x2, y2, score])

    # Update Tracker
    tracked_objects = tracker.update_tracks(people, frame=frame)
    people_count = len(tracked_objects)
    people_counts.append(people_count)

    # Draw bounding boxes
    frame = draw_boxes(frame, tracked_objects, people_count)

    # Display Output
    cv2.imshow("Crowd Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Train Anomaly Detector
features = preprocess_features(people_counts)  # Convert counts to features
anomaly_detector.fit(features)

# Detect Anomalies
anomaly_labels = anomaly_detector.predict(features)
anomalous_frames = [i for i, label in enumerate(anomaly_labels) if label == -1]

print(f"Anomalous frames detected at: {anomalous_frames}")

cap.release()
cv2.destroyAllWindows()
