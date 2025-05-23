import os
import glob
import logging
import numpy as np
from PIL import Image
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assemble_model(models_dir):
    """Combine split ONNX files into single model file"""
    model_name = "wound_model.onnx"
    model_path = os.path.join(models_dir, model_name)

    if os.path.exists(model_path):
        logger.info(f"Found existing model at {model_path}")
        return model_path

    parts_pattern = os.path.join(models_dir, "wound_model_v26.onnx.part*")
    parts = sorted(glob.glob(parts_pattern),
                   key=lambda x: int(x.split("part")[-1]))

    if not parts:
        raise FileNotFoundError(f"No model parts found in {models_dir}")

    logger.info(f"Assembling model from {len(parts)} parts...")

    try:
        with open(model_path, "wb") as outfile:
            for part in parts:
                logger.info(f"Adding {os.path.basename(part)}")
                with open(part, "rb") as infile:
                    outfile.write(infile.read())

        logger.info(f"Model assembly complete: {model_path}")
        return model_path

    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)
        logger.error(f"Assembly failed: {str(e)}")
        raise

def preprocess_image(img: Image.Image) -> np.ndarray:
    """EXACT preprocessing from training pipeline"""
    # Convert to array and resize
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)

    # Replicate keras.applications.efficientnet_v2.preprocess_input
    img_array = img_array[..., ::-1]  # RGB -> BGR
    img_array = (img_array - 91.4953) / (255 - 91.4953)  # Original model preprocessing

    # Add batch dimension
    return np.expand_dims(img_array, axis=0)


def predict_image(file_stream, session: ort.InferenceSession):
    """Process image and return predictions"""
    try:
        img = Image.open(file_stream).convert('RGB')
        input_tensor = preprocess_image(img)

        # Get input details
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})

        # Format results
        class_names = [
            "abrasion", "acne", "actinic keratosis", "avulsions",
            "bruise", "bugbite", "burn", "cellulitis", "chickenpox",
            "cut", "DFU", "ingrown nails", "measles", "monkeypox",
            "normal", "pressure wounds", "puncture", "rosacea",
            "venous wounds", "warts"
        ]

        probabilities = outputs[0][0]
        return sorted([
            {
                "class": name,
                "confidence": f"{prob * 100:.2f}%",
                "probability": float(prob)
            }
            for name, prob in zip(class_names, probabilities)
            if prob >= 0.01
        ], key=lambda x: -x['probability'])[:5]

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise