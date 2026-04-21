from flask import Flask, request, render_template
import pickle
import numpy as np
import cv2

app = Flask(__name__)

model = pickle.load(open("final_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Read image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (50, 50))
    image = image.flatten().reshape(1, -1)

    prediction = model.predict(image)

    if prediction[0] == 1:
        result = "Flood Detected 🌊"
    else:
        result = "No Flood ❌"

    return result

if __name__ == "__main__":
    import os
    port =int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0' ,port=port)
