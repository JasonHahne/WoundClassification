import os
from flask import Flask, request, render_template, jsonify
from utils import assemble_model, predict_image
import onnxruntime as ort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Initialize model with error handling
try:
    # Verify models directory exists
    if not os.path.exists('models'):
        raise FileNotFoundError("models directory not found")

    # Assemble and load model
    model_path = assemble_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Assembled model not found at {model_path}")

    app.config['MODEL_SESSION'] = ort.InferenceSession(model_path)
    app.logger.info(f"Successfully loaded model from {model_path}")

except Exception as e:
    app.logger.error(f"Critical initialization error: {str(e)}")
    raise  # Fail fast if model can't load


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        result = predict_image(file.stream)
        return jsonify({
            "success": True,
            "predictions": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)