import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for speed

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def detect_people(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO object detection
        results = model(frame)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Check if detected class is 'person' (COCO class ID 0)
                if cls == 0 and conf > 0.5:
                    detections.append(([x1, y1, x2, y2], conf, cls))

        # Track detected people
        tracks = tracker.update_tracks(detections, frame=frame)

        people_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            people_count += 1
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, f"Crowd Count: {people_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow('People Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_people("dataset/crowd_video.mp4", "outputs/detected_people.avi")
