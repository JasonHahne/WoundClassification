import os
import logging
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from utils import assemble_model, predict_image
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Initialize model once
try:
    logger.info("Initializing model...")
    models_dir = Path("models")
    model_path = assemble_model(models_dir)

    # Configure ONNX runtime
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    # Create session
    app.config['MODEL_SESSION'] = ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=['CPUExecutionProvider']
    )

    # Verify input shape
    input_shape = app.config['MODEL_SESSION'].get_inputs()[0].shape
    logger.info(f"Model input shape: {input_shape}")

except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise RuntimeError("Failed to initialize model") from e


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if '.' not in file.filename or ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Use PNG, JPG, or JPEG"}), 400

        # Process image
        file.stream.seek(0)
        result = predict_image(file.stream, app.config['MODEL_SESSION'])
        return jsonify({"success": True, "predictions": result})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "Analysis failed. Please try again."}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)