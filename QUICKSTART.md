# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

Create folders and add your images:

```
dataset/
├── train/
│   ├── defective/  ← Put defective bearing images here
│   └── ok/         ← Put OK bearing images here
└── test/
    ├── defective/  ← Test defective images
    └── ok/         ← Test OK images
```

**Minimum:** 50 images per class for training (more is better!)

### Step 3: Train the Model

```bash
# Basic training (Custom CNN)
python train.py

# Or use transfer learning (faster, often better)
python train.py --model_type transfer --epochs 30
```

Training will save the best model to `models/best_model.h5`

### Step 4: Run the Web App

```bash
python app.py
```

Open your browser: `http://localhost:5000`

### Step 5: Test Prediction

Upload an image through the web interface or use CLI:

```bash
python predict.py path/to/your/image.jpg
```

## 📋 Common Commands

```bash
# Train custom CNN
python train.py --model_type custom --epochs 50

# Train with transfer learning
python train.py --model_type transfer --epochs 30

# Train with fine-tuning
python train.py --model_type transfer --fine_tune

# Predict single image
python predict.py image.jpg

# Generate Grad-CAM visualization
python gradcam.py image.jpg --output_path result.jpg
```

## ⚠️ Troubleshooting

**No model found?** → Train first: `python train.py`

**Dataset not found?** → Create folder structure as shown above

**Out of memory?** → Reduce batch size: `--batch_size 16`

**Slow training?** → Use transfer learning: `--model_type transfer`

## 🎯 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with different model types and hyperparameters
- Add more training data to improve accuracy
- Deploy to production (Render, Railway, Heroku)

---

**Happy Training! 🎉**
