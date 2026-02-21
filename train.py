"""
Training script for Ball Bearing Defect Detection CNN Model
Supports both custom CNN and transfer learning approaches
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from utils import get_data_generators, get_test_generator, check_dataset_exists, ensure_dir


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=1):
    """
    Build a custom CNN model for defect detection
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
        
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def build_transfer_learning_model(base_model_name='mobilenet', 
                                 input_shape=(224, 224, 3), 
                                 num_classes=1,
                                 fine_tune=False):
    """
    Build a transfer learning model using pre-trained weights
    
    Args:
        base_model_name: Name of base model ('mobilenet' or 'resnet')
        input_shape: Input image shape
        num_classes: Number of output classes
        fine_tune: Whether to fine-tune base model layers
        
    Returns:
        model: Compiled Keras model
    """
    # Load base model
    if base_model_name.lower() == 'mobilenet':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'resnet':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers initially
    base_model.trainable = not fine_tune
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
        
    Returns:
        model: Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model


def train_model(model_type='custom',
               epochs=50,
               batch_size=32,
               learning_rate=0.001,
               img_size=(224, 224),
               data_dir='dataset',
               model_save_dir='models',
               fine_tune=False):
    """
    Train the CNN model
    
    Args:
        model_type: Type of model ('custom' or 'transfer')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        img_size: Image size (height, width)
        data_dir: Dataset directory path
        model_save_dir: Directory to save models
        fine_tune: Whether to fine-tune (for transfer learning)
        
    Returns:
        history: Training history
        model: Trained model
    """
    print("=" * 60)
    print("BALL BEARING DEFECT DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Check dataset
    if not check_dataset_exists(data_dir):
        print("\nERROR: Dataset not found!")
        print(f"Please organize your dataset as follows:")
        print(f"{data_dir}/")
        print(f"  train/")
        print(f"    defective/")
        print(f"    ok/")
        print(f"  test/")
        print(f"    defective/")
        print(f"    ok/")
        return None, None
    
    # Ensure model directory exists
    ensure_dir(model_save_dir)
    
    # Get data generators
    print("\nLoading data generators...")
    train_gen, val_gen, class_weights = get_data_generators(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        augmentation=True
    )
    
    if train_gen is None:
        return None, None
    
    print(f"Found {train_gen.samples} training samples")
    print(f"Found {val_gen.samples} validation samples")
    print(f"Class weights: {class_weights}")
    
    # Build model
    print(f"\nBuilding {model_type} model...")
    input_shape = (*img_size, 3)
    
    if model_type == 'custom':
        model = build_custom_cnn(input_shape=input_shape)
    elif model_type == 'transfer':
        base_model = 'mobilenet'  # Can be changed to 'resnet'
        model = build_transfer_learning_model(
            base_model_name=base_model,
            input_shape=input_shape,
            fine_tune=fine_tune
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile model
    model = compile_model(model, learning_rate=learning_rate)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    model_path = os.path.join(model_save_dir, 'best_model.h5')
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_gen = get_test_generator(data_dir=data_dir, img_size=img_size, batch_size=batch_size)
    
    if test_gen is not None:
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_gen, verbose=1)
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        
        # Generate predictions for confusion matrix
        test_gen.reset()
        predictions = model.predict(test_gen)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        class_names = ['Defective', 'OK']
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(model_save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")
        plt.close()
    
    # Plot training history
    plot_training_history(history, model_save_dir)
    
    return history, model


def plot_training_history(history, save_dir='models'):
    """
    Plot training history (accuracy and loss)
    
    Args:
        history: Training history object
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Ball Bearing Defect Detection Model')
    parser.add_argument('--model_type', type=str, default='custom',
                       choices=['custom', 'transfer'],
                       help='Model type: custom or transfer')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (square)')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Dataset directory path')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune base model (for transfer learning)')
    
    args = parser.parse_args()
    
    # Train model
    history, model = train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=(args.img_size, args.img_size),
        data_dir=args.data_dir,
        fine_tune=args.fine_tune
    )
    
    if history is not None:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
