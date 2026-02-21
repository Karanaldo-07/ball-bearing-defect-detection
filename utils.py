"""
Utility functions for Ball Bearing Defect Detection
Handles data loading, preprocessing, and helper functions
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')


def check_dataset_exists(base_path='dataset'):
    """
    Check if dataset directory exists and has required structure
    
    Args:
        base_path: Base path to dataset directory
        
    Returns:
        bool: True if dataset exists, False otherwise
    """
    train_defective = os.path.join(base_path, 'train', 'defective')
    train_ok = os.path.join(base_path, 'train', 'ok')
    test_defective = os.path.join(base_path, 'test', 'defective')
    test_ok = os.path.join(base_path, 'test', 'ok')
    
    required_dirs = [train_defective, train_ok, test_defective, test_ok]
    return all(os.path.exists(d) and os.path.isdir(d) for d in required_dirs)


def get_data_generators(data_dir='dataset', 
                       img_size=(224, 224), 
                       batch_size=32,
                       validation_split=0.2,
                       augmentation=True):
    """
    Create data generators for training and validation
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size (height, width)
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        augmentation: Whether to apply data augmentation
        
    Returns:
        tuple: (train_generator, validation_generator, class_weights)
    """
    train_dir = os.path.join(data_dir, 'train')
    
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=validation_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Check if dataset exists
    if not check_dataset_exists(data_dir):
        print(f"Warning: Dataset not found at {data_dir}")
        print("Creating dummy data generators. Please add your dataset to proceed.")
        return None, None, None
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # Calculate class weights for imbalanced datasets
    class_weights = calculate_class_weights(train_generator)
    
    return train_generator, validation_generator, class_weights


def calculate_class_weights(generator):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        generator: Training data generator
        
    Returns:
        dict: Class weights dictionary
    """
    try:
        classes = generator.classes
        unique_classes = np.unique(classes)
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=classes
        )
        class_weights_dict = dict(zip(unique_classes, weights))
        return class_weights_dict
    except Exception as e:
        print(f"Warning: Could not calculate class weights: {e}")
        return {0: 1.0, 1: 1.0}


def get_test_generator(data_dir='dataset', img_size=(224, 224), batch_size=32):
    """
    Create data generator for testing
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size (height, width)
        batch_size: Batch size for testing
        
    Returns:
        test_generator: Test data generator
    """
    test_dir = os.path.join(data_dir, 'test')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory not found at {test_dir}")
        return None
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator


def preprocess_image(image_path, img_size=(224, 224)):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image file
        img_size: Target image size (height, width)
        
    Returns:
        numpy array: Preprocessed image array
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, img_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def get_image_info(image_path):
    """
    Get basic information about an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Image information
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        
        return {
            'height': height,
            'width': width,
            'channels': channels,
            'size_mb': os.path.getsize(image_path) / (1024 * 1024)
        }
    except Exception as e:
        print(f"Error getting image info: {e}")
        return None


def ensure_dir(directory):
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def ensure_model_built(model, input_shape=(224, 224, 3)):
    """
    Ensure a Keras model has defined inputs/outputs by calling it once.

    This prevents errors like:
    "The layer sequential has never been called and thus has no defined output."

    Args:
        model: A Keras model instance
        input_shape: (H, W, C) used for dummy forward pass

    Returns:
        The same model instance (built/called)
    """
    if model is None:
        raise ValueError("Model is None")

    # If outputs are already defined, we're done.
    try:
        outs = model.outputs
        if outs is not None and len(outs) > 0:
            return model
    except Exception:
        pass

    h, w, c = input_shape
    dummy = np.zeros((1, h, w, c), dtype=np.float32)

    # Prefer predict() to match common Keras usage.
    try:
        model.predict(dummy, verbose=0)
        return model
    except Exception:
        # Fallback: direct call for some model types.
        try:
            _ = model(dummy, training=False)
            return model
        except Exception as e:
            raise RuntimeError(f"Could not build model with dummy input: {e}")
