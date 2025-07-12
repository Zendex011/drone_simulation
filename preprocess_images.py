import os
import cv2
import numpy as np

# Original dataset path
DATASET_DIR = "./object_tracking_dataset"
PROCESSED_DIR = "./processed_dataset"

IMG_SIZE = 128  # Target size for images

def resize_and_save_images(input_dir, output_dir):
    """Resizes images to IMG_SIZE x IMG_SIZE and saves them in output_dir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = os.listdir(input_dir)
    
    for category in categories:
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Skipping invalid file: {img_path}")
                continue

            # Resize image
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Save processed image
            save_path = os.path.join(output_category_path, img_name)
            cv2.imwrite(save_path, img_resized)

    print(f"✅ Images saved in {output_dir}")

# Process both train and validation sets
resize_and_save_images(os.path.join(DATASET_DIR, "train"), os.path.join(PROCESSED_DIR, "train"))
resize_and_save_images(os.path.join(DATASET_DIR, "val"), os.path.join(PROCESSED_DIR, "val"))
