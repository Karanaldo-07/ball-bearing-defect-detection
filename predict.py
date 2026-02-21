"""
Prediction module for Ball Bearing Defect Detection
Handles model loading and inference
"""

import os
import numpy as np
from tensorflow import keras
from utils import preprocess_image, get_image_info


class BearingDefectPredictor:
    """
    Class for making predictions on ball bearing images
    """
    
    def __init__(self, model_path='models/best_model.h5'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model from file
        """
        try:
            if not os.path.exists(self.model_path):
                # Try alternative paths
                alt_paths = [
                    'models/final_model.h5',
                    os.path.join(os.path.dirname(__file__), 'models', 'best_model.h5'),
                    os.path.join(os.path.dirname(__file__), 'models', 'final_model.h5')
                ]
                
                model_loaded = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.model_path = alt_path
                        self.model = keras.models.load_model(alt_path)
                        model_loaded = True
                        print(f"Model loaded from {alt_path}")
                        break
                
                if not model_loaded:
                    raise FileNotFoundError(
                        f"Model not found at {self.model_path}. "
                        "Please train the model first using train.py"
                    )
            else:
                self.model = keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
        
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def predict(self, image_path, return_confidence=True):
        """
        Predict defect status for an image
        
        Args:
            image_path: Path to image file
            return_confidence: Whether to return confidence score
            
        Returns:
            dict: Prediction results with class and confidence
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize the predictor.")
        
        # Validate image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get image info
        img_info = get_image_info(image_path)
        if img_info is None:
            raise ValueError(f"Invalid image file: {image_path}")
        
        # Preprocess image
        try:
            processed_img = preprocess_image(image_path, img_size=self.img_size)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
        
        # Make prediction
        try:
            prediction = self.model.predict(processed_img, verbose=0)
            confidence = float(prediction[0][0])
            
            # Binary classification: 0 = Defective, 1 = OK
            # Sigmoid output: < 0.5 = Defective, >= 0.5 = OK
            if confidence < 0.5:
                predicted_class = 'Defective'
                defect_confidence = 1 - confidence
            else:
                predicted_class = 'OK'
                defect_confidence = confidence
            
            result = {
                'class': predicted_class,
                'confidence': float(defect_confidence),
                'raw_score': float(confidence),
                'image_info': img_info
            }
            
            if return_confidence:
                return result
            else:
                return predicted_class
        
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
    
    def predict_batch(self, image_paths):
        """
        Predict defect status for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            list: List of prediction results
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'class': 'Error',
                    'confidence': 0.0,
                    'error': str(e),
                    'image_path': img_path
                })
        return results


def predict_image(image_path, model_path='models/best_model.h5'):
    """
    Convenience function for single image prediction
    
    Args:
        image_path: Path to image file
        model_path: Path to model file
        
    Returns:
        dict: Prediction results
    """
    predictor = BearingDefectPredictor(model_path=model_path)
    return predictor.predict(image_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Ball Bearing Defect')
    parser.add_argument('image_path', type=str,
                       help='Path to image file')
    parser.add_argument('--model_path', type=str, default='models/best_model.h5',
                       help='Path to model file')
    
    args = parser.parse_args()
    
    try:
        predictor = BearingDefectPredictor(model_path=args.model_path)
        result = predictor.predict(args.image_path)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Image: {args.image_path}")
        print(f"Predicted Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Raw Score: {result['raw_score']:.4f}")
        if 'image_info' in result:
            info = result['image_info']
            print(f"Image Size: {info['width']}x{info['height']}")
            print(f"Channels: {info['channels']}")
        print("=" * 60)
    
    except Exception as e:
        print(f"Error: {e}")
