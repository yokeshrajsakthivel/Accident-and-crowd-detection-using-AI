import cv2
import numpy as np

# Draw Bounding Boxes
def draw_boxes(frame, tracked_objects, count):
    for obj in tracked_objects:
        if not obj.is_confirmed():
            continue
        x1, y1, x2, y2 = obj.to_ltwh()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj.track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Crowd Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Convert People Count Data to Feature Vector
def preprocess_features(counts):
    return np.array(counts).reshape(-1, 1)
