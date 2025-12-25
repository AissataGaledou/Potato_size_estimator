# test.py — for binary (potato vs non-potato)
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("mobilenetv2_potato_binary.keras")

# Load and preprocess image
if len(sys.argv) < 2:
    print("Usage: python3 test.py <image_path>")
    sys.exit()

image_path = sys.argv[1]
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Could not read image: {image_path}")
    sys.exit()

img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)[0][0]
label = "Potato " if pred >= 0.5 else "Not Potato 🚫"

print(f" Prediction: {label} (confidence: {pred:.2f})")

