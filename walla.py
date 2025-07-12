import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("object_tracker_model.h5")
CATEGORIES = ["mechanical_pen", "pen", "pencil"]
IMG_SIZE = 128

def predict_object_from_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)
    predicted_class = CATEGORIES[np.argmax(prediction)]
    return predicted_class

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    predicted_class = predict_object_from_frame(frame)
    cv2.putText(display_frame, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Prediction Test", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
