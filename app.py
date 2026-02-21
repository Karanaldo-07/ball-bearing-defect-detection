"""
Flask Web Application for Ball Bearing Defect Detection
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from predict import BearingDefectPredictor
from gradcam import GradCAM
from utils import ensure_dir, ensure_model_built
import cv2
import numpy as np
from tensorflow import keras
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Initialize predictor (will be loaded on first use)
predictor = None
gradcam_model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_predictor():
    """Get or initialize predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = BearingDefectPredictor()
        except Exception as e:
            print(f"Warning: Could not load predictor: {e}")
            predictor = None
    return predictor


def get_gradcam_model():
    """Get or initialize Grad-CAM model"""
    global gradcam_model
    if gradcam_model is None:
        try:
            model_paths = [
                'models/best_model.h5',
                'models/final_model.h5'
            ]
            for model_path in model_paths:
                if os.path.exists(model_path):
                    gradcam_model = keras.models.load_model(model_path)
                    # Warm up the model once to ensure inputs/outputs are defined
                    ensure_model_built(gradcam_model, input_shape=(224, 224, 3))
                    break
        except Exception as e:
            print(f"Warning: Could not load model for Grad-CAM: {e}")
            gradcam_model = None
    return gradcam_model


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP, WEBP)')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        ensure_dir(app.config['UPLOAD_FOLDER'])
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            flash('Model not found. Please train the model first using train.py')
            return redirect(url_for('index'))
        
        # Make prediction
        result = pred.predict(filepath)
        
        # Generate Grad-CAM visualization if requested
        gradcam_path = None
        if request.form.get('show_gradcam') == 'true':
            try:
                model = get_gradcam_model()
                if model is not None:
                    gradcam = GradCAM(model)
                    gradcam_filename = f"gradcam_{filename}"
                    gradcam_path = os.path.join('static', 'gradcam', gradcam_filename)
                    ensure_dir(os.path.dirname(gradcam_path))
                    gradcam.save_visualization(filepath, gradcam_path)
                    gradcam_path = f"static/gradcam/{gradcam_filename}"
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
        
        # Prepare result data
        result_data = {
            'filename': filename,
            'image_path': f"uploads/{filename}",
            'image_url': url_for('uploaded_file', filename=filename),
            'predicted_class': result['class'],
            'confidence': result['confidence'],
            'raw_score': result['raw_score'],
            'gradcam_path': gradcam_path,
            'image_info': result.get('image_info', {})
        }
        
        return render_template('result.html', result=result_data)
    
    except Exception as e:
        flash(f'Error during prediction: {str(e)}')
        return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        ensure_dir(app.config['UPLOAD_FOLDER'])
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            return jsonify({'error': 'Model not found'}), 500
        
        # Make prediction
        result = pred.predict(filepath)
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': result['class'],
                'confidence': result['confidence'],
                'raw_score': result['raw_score']
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    """Health check endpoint"""
    pred = get_predictor()
    model_status = 'loaded' if pred is not None else 'not_loaded'
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    flash('An error occurred. Please try again.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Ensure necessary directories exist
    ensure_dir(app.config['UPLOAD_FOLDER'])
    ensure_dir('static/gradcam')
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("BALL BEARING DEFECT DETECTION - WEB APPLICATION")
    print("=" * 60)
    print(f"Starting server on port {port}")
    print(f"Debug mode: {debug}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
