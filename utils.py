import os
import glob
import logging
import numpy as np
from PIL import Image
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assemble_model():
    """Combine split ONNX files into single model file"""
    model_name = "assembled_model.onnx"
    parts_pattern = os.path.join("models", "model.onnx.part*")

    # Check if model already exists
    if os.path.exists(model_name):
        logger.info("Found existing assembled model")
        return model_name

    logger.info("Assembling model from parts...")

    # Get sorted list of parts
    parts = sorted(glob.glob(parts_pattern),
                   key=lambda x: int(x.split("part")[-1]))

    # Check for missing parts
    if not parts:
        raise FileNotFoundError("No model parts found in models/ directory")

    try:
        total_size = sum(os.path.getsize(p) for p in parts)
        logger.info(f"Found {len(parts)} parts, total size: {total_size / 1024 / 1024:.2f}MB")

        # Assemble parts
        with open(model_name, "wb") as outfile:
            for part in parts:
                logger.debug(f"Adding {part}")
                with open(part, "rb") as infile:
                    outfile.write(infile.read())

        logger.info(f"Successfully assembled {model_name}")
        return model_name

    except Exception as e:
        # Clean up if assembly failed
        if os.path.exists(model_name):
            os.remove(model_name)
        logger.error(f"Model assembly failed: {str(e)}")
        raise


def predict_image(file_stream):
    """Process image and run prediction"""
    try:
        # Load image
        img = Image.open(file_stream).convert('RGB')

        # Preprocess
        img = img.resize((224, 224))  # Match model input size
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array.transpose(2, 0, 1), axis=0)  # CHW format

        # Run inference
        session = ort.InferenceSession("assembled_model.onnx")
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)

        # Process outputs
        class_names = [
            "abrasion", "acne", "actinic keratosis", "avulsions",
            "bruise", "bugbite", "burn", "cellulitis", "chickenpox",
            "cut", "DFU", "ingrown nails", "measles", "monkeypox",
            "normal", "pressure wounds", "puncture", "rosacea",
            "venous wounds", "warts"
        ]

        probabilities = outputs[0][0]
        results = []

        for i, prob in enumerate(probabilities):
            if prob >= 0.01:  # 1% confidence threshold
                results.append({
                    "class": class_names[i],
                    "confidence": f"{prob * 100:.2f}%",
                    "probability": float(prob)
                })

        # Sort and return top 5
        return sorted(results, key=lambda x: -x['probability'])[:5]

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise