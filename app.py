import os
from flask import Flask, request, render_template, jsonify
from utils import assemble_model, predict_image
import onnxruntime as ort

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
app.secret_key = os.urandom(24)

# Initialize model
try:
    if not os.path.exists('models'):
        raise FileNotFoundError("Models directory not found")

    model_path = assemble_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Assembled model not found at {model_path}")

    app.config['MODEL_SESSION'] = ort.InferenceSession(model_path)
    app.logger.info(f"Model loaded successfully: {os.path.getsize(model_path) / 1024 / 1024:.2f}MB")

except Exception as e:
    app.logger.error(f"Initialization failed: {str(e)}")
    raise


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({"error": "Allowed formats: PNG, JPG, JPEG"}), 400

        result = predict_image(file.stream)
        return jsonify({
            "success": True,
            "predictions": result
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Analysis failed. Please try another image."
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)