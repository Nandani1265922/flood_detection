import cv2
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

data = []
labels = []

# Flood images (label = 1)
# for img in os.listdir("dataset/flood"):
flood_path=r"D:\Flood_Detection_Project\dataset\flood"
print("Loading flood images...")
for img in os.listdir(r"D:\Flood_Detection_Project\dataset\flood"):
    path = os.path.join(r"D:\Flood_Detection_Project\dataset\flood", img)
    image = cv2.imread(path)
    if image is not None:
        image=cv2.resize(image,(50,50))
        data.append(image.flatten())
        labels.append(1)


print(f"flood images loaded :{labels.count(1)}")

# No flood images (label = 0)
# for img in os.listdir("dataset/no_flood"):
no_flood_path=r"D:\Flood_Detection_Project\dataset\no_flood"
print("loading no_flood images...")
for img in os.listdir(r"D:\Flood_Detection_Project\dataset\no_flood"):
    path = os.path.join(r"D:\Flood_Detection_Project\dataset\no_flood", img)
    image = cv2.imread(path)
    if image is not None:
        image = cv2.resize(image, (50, 50))
        data.append(image.flatten())
        labels.append(0)


print(f"No_flood images loaded : {labels.count(0)}")

if len(data) ==0:
    print("error no images found check folder path")
else:
# Convert to array
   X = np.array(data)
   y = np.array(labels)

# Train model
   print("Flood detection model trained!")
   model = LogisticRegression(max_iter=1000)
   model.fit(X, y)

# Save model
   print("saving model to model.pkl")
   with open("final_model.pkl", "wb") as f:
       pickle.dump(model,f)

#    print("Flood detection model trained!")
#    print("total imagesn", len(data))
print("done! check your folder now")