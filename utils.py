"""
Utility functions for Ball Bearing Defect Detection
Deployment-safe version (NO TensorFlow)
"""

import os
import numpy as np
import cv2


# -------------------------
# Ensure directory exists
# -------------------------
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(image_path, img_size=(224, 224)):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# -------------------------
# Image info
# -------------------------
def get_image_info(image_path):

    try:
        img = cv2.imread(image_path)

        if img is None:
            return None

        h, w = img.shape[:2]

        return {
            "height": h,
            "width": w,
            "channels": img.shape[2] if len(img.shape) == 3 else 1,
            "size_mb": os.path.getsize(image_path) / (1024 * 1024),
        }

    except Exception as e:
        print(f"Image info error: {e}")
        return None


# -------------------------
# Dummy function (kept for compatibility)
# -------------------------
def ensure_model_built(model, input_shape=(224, 224, 3)):
    """
    Not needed for TFLite.
    Kept only to avoid import errors.
    """
    return model