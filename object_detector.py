# object_detector.py

import cv2
import numpy as np
import tensorflow as tf

# ==== Load CNN Model ====
model = tf.keras.models.load_model("object_tracker_model.h5")
CATEGORIES = ["mechanical_pen", "pen", "pencil"]
IMG_SIZE = 128

def predict_object_from_frame(frame):
   
    try:
        # Crop center of the frame (central square region)
        h, w, _ = frame.shape
        min_dim = min(h, w)
        center_crop = frame[h//2 - min_dim//2:h//2 + min_dim//2, w//2 - min_dim//2:w//2 + min_dim//2]

        img = cv2.resize(center_crop, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)
        confidence = np.max(prediction)
        predicted_class = CATEGORIES[np.argmax(prediction)]

        print(f"[DEBUG] Predicted: {predicted_class}, Confidence: {confidence:.2f}")

        if confidence > 0.7:  # You can lower this if needed
            return predicted_class
        else:
            return None
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
