"""
Flask Web Application for Ball Bearing Defect Detection
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from predict import BearingDefectPredictor
from gradcam import GradCAM
from utils import ensure_dir, ensure_model_built
import os

import gdown

MODEL_PATH = "models/best_model.h5"

# -------------------------
# Lazy model download
# -------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?id=1SRnyn7LkFBTArmX9vhg2mwTDT2QRazW"
        gdown.download(url, MODEL_PATH, quiet=False)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Lazy globals
predictor = None
gradcam_model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------
# Lazy predictor loader
# -------------------------
def get_predictor():
    global predictor

    if predictor is None:
        try:
            ensure_model()   # download only when needed
            predictor = BearingDefectPredictor()
        except Exception as e:
            print(f"Predictor load failed: {e}")
            predictor = None

    return predictor


# -------------------------
# Lazy GradCAM model
# -------------------------
def get_gradcam_model():
    global gradcam_model

    if gradcam_model is None:
        try:
            ensure_model()

            # Import tensorflow ONLY when needed (important for RAM)
            from tensorflow import keras

            if os.path.exists(MODEL_PATH):
                gradcam_model = keras.models.load_model(MODEL_PATH)
                ensure_model_built(gradcam_model, input_shape=(224, 224, 3))

        except Exception as e:
            print(f"GradCAM load failed: {e}")
            gradcam_model = None

    return gradcam_model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)

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

        gradcam_path = None

        if request.form.get('show_gradcam') == 'true':
            try:
                model = get_gradcam_model()

                if model is not None:
                    gradcam = GradCAM(model)

                    gradcam_filename = f"gradcam_{filename}"
                    gradcam_path_full = os.path.join('static', 'gradcam', gradcam_filename)

                    ensure_dir(os.path.dirname(gradcam_path_full))

                    gradcam.save_visualization(filepath, gradcam_path_full)

                    gradcam_path = f"static/gradcam/{gradcam_filename}"

            except Exception as e:
                print(f"GradCAM error: {e}")

        result_data = {
            'filename': filename,
            'image_url': url_for('uploaded_file', filename=filename),
            'predicted_class': result['class'],
            'confidence': result['confidence'],
            'raw_score': result['raw_score'],
            'gradcam_path': gradcam_path,
            'image_info': result.get('image_info', {})
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
    ensure_dir('static/gradcam')

    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print("Starting server...")

    app.run(host='0.0.0.0', port=port, debug=debug)