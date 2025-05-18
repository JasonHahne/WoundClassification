from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
from utils import predict_image

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-123')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save file
            file.save(filepath)

            # Get predictions
            result = predict_image(filepath)

            # Cleanup
            try:
                os.remove(filepath)
            except:
                pass

            if result['success']:
                return render_template('index.html',
                                       predictions=result['predictions'],
                                       uploaded_image=filename)
            return render_template('index.html', error=result['error'])

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

# Run the application directly only in development
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=10000, debug=True)  # Added debug flag