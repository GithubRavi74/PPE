from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO("models/best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    detections = results[0].boxes.data.tolist()

    return jsonify({
        "detections": detections
    })

if __name__ == "__main__":
    app.run(debug=True)