import cv2
import numpy as np
import pickle
from detect_people import detect_people

# Load trained anomaly detection model
with open("models/anomaly_detector.pkl", "rb") as f:
    model = pickle.load(f)

def detect_anomalies(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated movement data (Replace with real tracking data)
        speed = np.random.uniform(1, 5)
        movement_variance = np.random.uniform(0.1, 2)
        direction_variance = np.random.uniform(0.05, 1)

        # Predict anomaly
        data_point = np.array([[speed, movement_variance, direction_variance]])
        anomaly_score = model.predict(data_point)

        # Mark frame if anomaly detected
        if anomaly_score[0] == -1:
            cv2.putText(frame, "Anomaly Detected!", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Anomaly Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_anomalies("outputs/detected_people.avi")
