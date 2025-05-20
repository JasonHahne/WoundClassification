import os
from flask import Flask, request, render_template, jsonify
from utils import assemble_model, predict_image
import onnxruntime as ort

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Initialize model directly (no longer using before_first_request)
try:
    model_path = assemble_model()
    app.config['MODEL_SESSION'] = ort.InferenceSession(model_path)
    app.logger.info("Model initialized successfully")
except Exception as e:
    app.logger.error(f"Model initialization failed: {str(e)}")
    raise

# Routes
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
    # Verify critical files
    required = ["models", "utils.py", "requirements.txt"]
    missing = [item for item in required if not os.path.exists(item)]

    if missing:
        raise FileNotFoundError(f"Missing required files/dirs: {missing}")

    # Start app
    app.run(host='0.0.0.0', port=10000)