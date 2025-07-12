import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
categories = ['mechanical_pen', 'pen', 'pencil']  # Match folder names

# Load model
model = tf.keras.models.load_model("object_tracker_model.h5")

# Test function
def test_image(path):
    img = cv2.imread(path)
    if img is None:
        print("⚠️ Image not found.")
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = categories[np.argmax(prediction)]
    print(f"🎯 Predicted: {label}")

# Change path to test different images
test_image("./pen_pencil_dataset/pencil/pencil (1).jpg")
