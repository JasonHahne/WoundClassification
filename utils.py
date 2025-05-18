import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Initialize model and classes
main_folder = os.path.join('app', 'data', 'images')
CLASSES = sorted(os.listdir(main_folder))

# Custom loss function
def focal_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    smoothing = 0.025
    y_true = y_true * (1 - smoothing) + smoothing / len(CLASSES)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(ce * weight)

# Load model
with keras.utils.custom_object_scope({'focal_loss': focal_loss, 'loss': focal_loss}):
    model = keras.models.load_model(os.path.join('app', 'models', 'data_v2-EfficientNetV2_v26_87.keras'))

def predict_image(image_path):
    try:
        # Preprocess image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image.astype('float32') / 255.0, axis=0)

        # Predict
        preds = model.predict(image)[0]

        # Process results
        MIN_CONFIDENCE = 0.01
        TOP_N = 5
        filtered = [
            {"class": CLASSES[i], "confidence": f"{p * 100:.2f}%", "probability": float(p)}
            for i, p in enumerate(preds) if p >= MIN_CONFIDENCE
        ]
        return {"success": True, "predictions": sorted(filtered, key=lambda x: -x['probability'])[:TOP_N]}

    except Exception as e:
        return {"success": False, "error": str(e)}