import cv2
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

data = []
labels = []

# Flood images (label = 1)
# for img in os.listdir("dataset/flood"):
for img in os.listdir(r"D:\Flood_Detection_Project\dataset\flood"):
    path = os.path.join(r"D:\Flood_Detection_Project\dataset\flood", img)
    image = cv2.imread(path)
    if image is not None:

        image=cv2.resize(image,(50,50))
    # image = cv2.resize(image, (50, 50))
        data.append(image.flatten())
        # data.append(image)
        labels.append(1)

# No flood images (label = 0)
# for img in os.listdir("dataset/no_flood"):
for img in os.listdir(r"D:\Flood_Detection_Project\dataset\no_flood"):
    path = os.path.join(r"D:\Flood_Detection_Project\dataset\no_flood", img)
    image = cv2.imread(path)
    if image is not None:

        image = cv2.resize(image, (50, 50))
        data.append(image.flatten())
        labels.append(0)

# Convert to array
X = np.array(data)
y = np.array(labels)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Flood detection model trained!")