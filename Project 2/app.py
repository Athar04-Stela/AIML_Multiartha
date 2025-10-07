from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model("cifar10_model.h5")

# Label CIFAR-10
labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((32,32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_id = np.argmax(pred)
    return jsonify({
        "class": labels[class_id],
        "confidence": float(np.max(pred))
    })

@app.route("/", methods=["GET"])
def home():
    return "CIFAR-10 Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
