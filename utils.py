import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

CLASSES = [
    "abrasion",
    "acne",
    "actinic keratosis",
    "avulsions",
    "bruise",
    "bugbite",
    "burn",
    "cellulitis",
    "chickenpox",
    "cut",
    "DFU",
    "ingrown nails",
    "measles",
    "monkeypox",
    "normal",
    "pressure wounds",
    "puncture",
    "rosacea",
    "venous wounds",
    "warts"
]

def focal_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    smoothing = 0.025
    y_true = y_true * (1 - smoothing) + smoothing / len(CLASSES)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(ce * weight)

# Load model once at startup
with keras.utils.custom_object_scope({'focal_loss': focal_loss}):
    model = keras.models.load_model(
        os.path.join('models', 'data_v2-EfficientNetV2_v26_87.keras'),
        compile=False
    )
    model.make_predict_function()  # For thread safety

def predict_image(file_stream):
    """Process image from file upload stream"""
    try:
        # Process in-memory
        image = Image.open(file_stream).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        preds = model.predict(image_array, verbose=0)[0]

        # Format results
        return {
            "success": True,
            "predictions": sorted([
                {"class": CLASSES[i], "confidence": f"{p*100:.2f}%"}
                for i, p in enumerate(preds) if p >= 0.01  # 1% threshold
            ], key=lambda x: -float(x['confidence'][:-1]))[:5],}  # Top 5
    except Exception as e:
        return {"success": False, "error": str(e)}