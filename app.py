"""
Flask Web Application for Ball Bearing Defect Detection (TFLite Version)
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from predict import BearingDefectPredictor
from utils import ensure_dir
import os
import requests

MODEL_PATH = "models/model.tflite"
MODEL_URL = "https://github.com/Karanaldo-07/ball-bearing-defect-detection/releases/download/model/model.tflite"


# -------------------------
# Download model if not exists
# -------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading model from GitHub Release...")

        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

        print("Model downloaded successfully.")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

predictor = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------
# Lazy predictor loader
# -------------------------
def get_predictor():
    global predictor

    if predictor is None:
        try:
            ensure_model()
            predictor = BearingDefectPredictor(MODEL_PATH)
        except Exception as e:
            print(f"Predictor load failed: {e}")
            predictor = None

    return predictor


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for('index'))

    try:
        ensure_dir(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        pred = get_predictor()

        if pred is None:
            flash('Model failed to load')
            return redirect(url_for('index'))

        result = pred.predict(filepath)

        result_data = {
            'filename': filename,
            'image_url': url_for('uploaded_file', filename=filename),
            'predicted_class': result['class'],
            'confidence': result['confidence'],
            'raw_score': result.get('raw_score', None)
        }

        return render_template('result.html', result=result_data)

    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        ensure_dir(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        pred = get_predictor()

        if pred is None:
            return jsonify({'error': 'Model load failed'}), 500

        result = pred.predict(filepath)

        return jsonify({
            'success': True,
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    pred = get_predictor()

    return jsonify({
        'status': 'healthy',
        'model_loaded': pred is not None
    })


@app.errorhandler(413)
def too_large(e):
    flash('File too large (max 16MB)')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    flash('Server error')
    return redirect(url_for('index'))


if __name__ == '__main__':

    ensure_dir(app.config['UPLOAD_FOLDER'])

    port = int(os.environ.get('PORT', 5000))

    print("Starting server...")

    app.run(host='0.0.0.0', port=port)