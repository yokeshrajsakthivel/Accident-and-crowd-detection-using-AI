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

# Define dataset paths
input_folder = "D:\VIT\ProjAi\src\dataset\images"
output_folder = "outputs"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

def detect_people_in_images():
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ No images found in dataset folder!")
        return

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load image
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Failed to load {filename}. Skipping...")
            continue

        # Run YOLOv8 object detection
        results = model(image)

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
        tracks = tracker.update_tracks(detections, frame=image)

        people_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            people_count += 1
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID {track.track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(image, f"People Count: {people_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save processed image
        cv2.imwrite(output_path, image)
        print(f"✅ Processed: {filename} -> Saved to {output_path}")

    print("\n🎉 All images processed successfully!")

if __name__ == "__main__":
    detect_people_in_images()
