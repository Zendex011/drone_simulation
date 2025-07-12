import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Path to dataset
dataset_path = "./pen_pencil_dataset"
IMG_SIZE = 128

# Prepare dataset
X, Y = [], []
categories = sorted(os.listdir(dataset_path))  # ['mechanical_pen', 'pen', 'pencil']
print("🗂️ Categories:", categories)

# Load images
for idx, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping: {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        X.append(img)
        Y.append(idx)

X = np.array(X)
Y = to_categorical(np.array(Y), num_classes=len(categories))

# Split data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"✅ Data loaded: {len(X_train)} train / {len(X_val)} val")

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_val, Y_val))

# Save
model.save("object_tracker_model.h5")
print("✅ Model saved.")
