import numpy as np
import pickle
from sklearn.ensemble import IsolationForest

# Simulated data: [speed, movement variance, direction variance]
normal_data = np.random.normal(loc=[2, 0.5, 0.2], scale=[0.5, 0.2, 0.1], size=(500, 3))
anomaly_data = np.random.normal(loc=[5, 2, 1], scale=[1, 0.5, 0.3], size=(50, 3))
data = np.vstack((normal_data, anomaly_data))

labels = np.array([1] * 500 + [-1] * 50)  # 1 = Normal, -1 = Anomaly

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(data)

# Save model
with open("models/anomaly_detector.pkl", "wb") as f:
    pickle.dump(model, f)

print("Anomaly detection model trained successfully!")
