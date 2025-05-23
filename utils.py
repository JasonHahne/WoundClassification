import os
import glob
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assemble_model(models_dir: Path) -> Path:
    """Combine split ONNX files into single model file"""
    model_name = "wound_model.onnx"
    model_path = models_dir / model_name

    if model_path.exists():
        logger.info(f"Found existing model at {model_path}")
        return model_path

    # Find all model parts
    parts_pattern = models_dir / "wound_model.onnx.part*"
    parts = sorted(glob.glob(str(parts_pattern)),
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
        if model_path.exists():
            model_path.unlink()
        logger.error(f"Assembly failed: {str(e)}")
        raise


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Replacement for efficientnet_v2.preprocess_input without TensorFlow"""
    # Convert to array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # Standard EfficientNetV2 preprocessing
    # Scale to [-1, 1] range
    img_array = (img_array - 0.5) * 2.0

    # Convert from HWC to CHW format
    return img_array.transpose(2, 0, 1)


def predict_image(file_stream, session: ort.InferenceSession):
    """Process image and return predictions using ONNX session"""
    try:
        # Load and resize image
        img = Image.open(file_stream).convert('RGB')
        img = img.resize((224, 224))

        # Preprocess without TensorFlow
        img_array = preprocess_image(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)

        # Format results (keep your existing class names)
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