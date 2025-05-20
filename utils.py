import os
import glob
import logging
import numpy as np
from PIL import Image
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assemble_model():
    """Combine split ONNX files into single model file"""
    model_name = "assembled_model.onnx"
    models_dir = "models"

    # Verify models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{os.path.abspath(models_dir)}' not found")

    parts_pattern = os.path.join(models_dir, "model.onnx.part*")
    parts = sorted(glob.glob(parts_pattern),
                   key=lambda x: int(x.split("part")[-1]))

    if not parts:
        available = "\n".join(os.listdir(models_dir))
        raise FileNotFoundError(
            f"No model parts found in {models_dir}\n"
            f"Directory contents:\n{available}"
        )

    # Assembly process
    try:
        logger.info(f"Found {len(parts)} model parts")

        with open(model_name, "wb") as outfile:
            for part in parts:
                logger.info(f"Adding {os.path.basename(part)}")
                with open(part, "rb") as infile:
                    outfile.write(infile.read())

        logger.info(f"Model assembly complete: {model_name}")
        return model_name

    except Exception as e:
        if os.path.exists(model_name):
            os.remove(model_name)
        logger.error(f"Assembly failed: {str(e)}")
        raise


def predict_image(file_stream):
    """Process image and return predictions"""
    try:
        # Load and preprocess image
        img = Image.open(file_stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array.transpose(2, 0, 1), axis=0)

        # Run inference
        session = ort.InferenceSession("assembled_model.onnx")
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)

        # Format results
        class_names = [
            "abrasion", "acne", "actinic keratosis", "avulsions",
            "bruise", "bugbite", "burn", "cellulitis", "chickenpox",
            "cut", "DFU", "ingrown nails", "measles", "monkeypox",
            "normal", "pressure wounds", "puncture", "rosacea",
            "venous wounds", "warts"
        ]

        return sorted([
            {
                "class": name,
                "confidence": f"{prob * 100:.2f}%",
                "probability": float(prob)
            }
            for name, prob in zip(class_names, outputs[0][0])
            if prob >= 0.01
        ], key=lambda x: -x['probability'])[:5]

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise