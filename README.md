# Ball Bearing Defect Detection System

A complete end-to-end deep learning project for detecting defects in ball bearing images using Convolutional Neural Networks (CNN). The system includes model training, prediction capabilities, and a web application for easy deployment.

## 🎯 Project Overview

This project uses deep learning to classify ball bearing images as either **Defective** or **OK**. It features:

- **Custom CNN Architecture** with BatchNorm and Dropout
- **Transfer Learning** support (MobileNetV2, ResNet50)
- **Data Augmentation** for improved generalization
- **Grad-CAM Visualization** for model interpretability
- **Flask Web Application** with modern UI
- **Production-ready** code with error handling

## 📁 Project Structure

```
ball_cursor/
├── dataset/
│   ├── train/
│   │   ├── defective/
│   │   └── ok/
│   └── test/
│       ├── defective/
│       └── ok/
├── models/              # Saved models
├── static/
│   ├── gradcam/        # Grad-CAM visualizations
│   └── style.css       # CSS styles
├── templates/
│   ├── index.html      # Home page
│   └── result.html     # Results page
├── uploads/            # Uploaded images
├── app.py              # Flask web application
├── train.py            # Model training script
├── predict.py          # Prediction module
├── gradcam.py          # Grad-CAM visualization
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- GPU (optional, but recommended for training)

### Step 1: Clone or Download Project

```bash
cd ball_cursor
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with TensorFlow, you can install CPU-only version:
```bash
pip install tensorflow-cpu
```

## 📊 Dataset Setup

### Option 1: Using Your Own Dataset

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── defective/      # Put defective bearing images here
│   └── ok/             # Put OK bearing images here
└── test/
    ├── defective/      # Test defective images
    └── ok/             # Test OK images
```

**Recommended:**
- Minimum 100 images per class for training
- Image formats: PNG, JPG, JPEG
- Consistent image sizes (will be resized to 224x224 automatically)

### Option 2: Using Public Datasets

You can use datasets like:
- [Kaggle Bearing Dataset](https://www.kaggle.com/datasets)
- [Mendeley Data](https://data.mendeley.com/)
- Create synthetic data using augmentation

## 🏋️ Training the Model

### Basic Training (Custom CNN)

```bash
python train.py
```

### Advanced Training Options

```bash
# Custom CNN with specific parameters
python train.py --model_type custom --epochs 50 --batch_size 32 --learning_rate 0.001

# Transfer Learning (MobileNetV2)
python train.py --model_type transfer --epochs 30 --batch_size 16

# Transfer Learning with Fine-tuning
python train.py --model_type transfer --epochs 50 --fine_tune

# ResNet50 Transfer Learning
# (Modify train.py to use 'resnet' as base_model_name)
```

### Training Parameters

- `--model_type`: `custom` or `transfer` (default: `custom`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--img_size`: Image size in pixels (default: 224)
- `--data_dir`: Dataset directory path (default: `dataset`)
- `--fine_tune`: Enable fine-tuning for transfer learning

### Training Outputs

After training, you'll find:
- `models/best_model.h5` - Best model based on validation loss
- `models/final_model.h5` - Final model after all epochs
- `models/training_history.png` - Accuracy and loss plots
- `models/confusion_matrix.png` - Confusion matrix visualization

## 🔮 Making Predictions

### Command Line Prediction

```bash
python predict.py path/to/image.jpg
```

### Using Python API

```python
from predict import BearingDefectPredictor

# Initialize predictor
predictor = BearingDefectPredictor(model_path='models/best_model.h5')

# Predict single image
result = predictor.predict('path/to/image.jpg')
print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Prediction

```python
from predict import BearingDefectPredictor

predictor = BearingDefectPredictor()
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

## 🎨 Grad-CAM Visualization

Generate Grad-CAM heatmaps to see which regions the model focuses on:

```bash
python gradcam.py path/to/image.jpg --output_path gradcam_result.jpg
```

Or use in Python:

```python
from gradcam import GradCAM
from tensorflow import keras

model = keras.models.load_model('models/best_model.h5')
gradcam = GradCAM(model)
gradcam.save_visualization('image.jpg', 'output.jpg')
```

## 🌐 Running the Web Application

### Start Flask Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface

1. **Upload Image**: Click "Choose File" or drag and drop an image
2. **Optional**: Check "Show Grad-CAM Visualization" for heatmap
3. **Analyze**: Click "Analyze Image"
4. **View Results**: See prediction, confidence score, and visualization

### API Endpoint

You can also use the REST API:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

Response:
```json
{
  "success": true,
  "prediction": {
    "class": "Defective",
    "confidence": 0.95,
    "raw_score": 0.05
  }
}
```

## 🚢 Deployment

### Local Deployment

The app runs on `localhost:5000` by default. For production:

1. Set `FLASK_DEBUG=False` in environment variables
2. Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Deploy to Render

1. Create a `render.yaml` or use Render dashboard
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Set environment variable: `PORT=5000`

### Deploy to Railway

1. Connect your GitHub repository
2. Railway will auto-detect Python
3. Set `PORT` environment variable
4. Deploy!

### Deploy to Heroku

1. Create `Procfile`:
```
web: python app.py
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## 📈 Model Performance Tips

### Improving Accuracy

1. **More Data**: Collect more training images (aim for 500+ per class)
2. **Data Augmentation**: Already enabled, but you can adjust in `utils.py`
3. **Transfer Learning**: Use `--model_type transfer` for better results
4. **Fine-tuning**: Enable `--fine_tune` after initial training
5. **Class Balance**: Ensure balanced dataset or use class weights (already implemented)
6. **Hyperparameter Tuning**: Adjust learning rate, batch size, epochs

### Handling Class Imbalance

The training script automatically calculates class weights. If you have severe imbalance:

1. Use data augmentation
2. Collect more samples for minority class
3. Use focal loss (modify `train.py`)

## 🐛 Troubleshooting

### Model Not Found Error

**Problem**: `Model not found at models/best_model.h5`

**Solution**: Train the model first using `python train.py`

### GPU Not Detected

**Problem**: Training is slow

**Solution**: 
- Install GPU version: `pip install tensorflow[and-cuda]`
- Or continue with CPU (slower but works)

### Out of Memory Error

**Solution**: 
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Reduce image size: `--img_size 128`
- Use transfer learning (smaller model)

### Dataset Not Found

**Problem**: `Dataset not found at dataset/`

**Solution**: 
- Create the folder structure as shown above
- Place your images in correct folders
- Or modify `data_dir` parameter

## 📝 Code Structure

### Key Files

- **`train.py`**: Model training with CNN and transfer learning
- **`predict.py`**: Prediction module with error handling
- **`gradcam.py`**: Grad-CAM visualization implementation
- **`utils.py`**: Data loading, preprocessing, utilities
- **`app.py`**: Flask web application with routes

### Model Architectures

1. **Custom CNN**: 4 conv blocks + dense layers
2. **MobileNetV2**: Lightweight, mobile-friendly
3. **ResNet50**: Deeper, more accurate (modify code to use)

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- TensorFlow/Keras team
- Flask community
- OpenCV contributors

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify dataset structure

---

**Built with ❤️ using TensorFlow, Keras, and Flask**
