"""
Grad-CAM visualization for Ball Bearing Defect Detection
Shows which regions of the image the model focuses on
"""

import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from utils import preprocess_image, ensure_model_built


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    
    def __init__(self, model, layer_name=None, input_shape=(224, 224, 3)):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to visualize
                       If None, uses the last convolutional layer
            input_shape: (H, W, C) for dummy warm-up call
        """
        self.model = ensure_model_built(model, input_shape=input_shape)
        self.layer_name = layer_name
        self.target_layer = None
        
        if layer_name is not None:
            self.target_layer = self.model.get_layer(layer_name)
            self.layer_name = self.target_layer.name
        else:
            self.target_layer = self._find_last_conv_layer(self.model)
            self.layer_name = self.target_layer.name if self.target_layer is not None else None
        
        if self.target_layer is None:
            raise ValueError("No convolutional layer found in the model")
        
        print(f"Using layer '{self.layer_name}' for Grad-CAM")

    def _find_last_conv_layer(self, model):
        """
        Recursively find the last Conv2D layer, including inside nested models.
        """
        for layer in reversed(model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Convolution2D)):
                return layer
            if isinstance(layer, keras.Model):
                nested = self._find_last_conv_layer(layer)
                if nested is not None:
                    return nested
        return None
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array
            pred_index: Class index to generate heatmap for (None for predicted class)
            
        Returns:
            heatmap: Grad-CAM heatmap
        """
        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        # Ensure built (some Keras setups need an explicit call)
        self.model = ensure_model_built(self.model, input_shape=tuple(img_array.shape[1:]))

        grad_model = keras.models.Model(
            self.model.inputs,
            [self.target_layer.output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def generate_visualization(self, image_path, img_size=(224, 224), alpha=0.4):
        """
        Generate Grad-CAM visualization for an image
        
        Args:
            image_path: Path to image file
            img_size: Image size for preprocessing
            alpha: Transparency factor for heatmap overlay
            
        Returns:
            tuple: (original_image, heatmap, superimposed_image)
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        original_img = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        img_array = preprocess_image(image_path, img_size=img_size)
        
        # Make prediction
        preds = self.model.predict(img_array, verbose=0)
        predicted_class = 'Defective' if preds[0][0] < 0.5 else 'OK'
        confidence = float(preds[0][0]) if predicted_class == 'OK' else float(1 - preds[0][0])
        
        # Generate heatmap
        try:
            heatmap = self.make_gradcam_heatmap(img_array)
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM heatmap: {e}")
            return None, None, None, predicted_class, confidence
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap_colored * alpha + img_rgb * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return original_img, heatmap, superimposed_img, predicted_class, confidence
    
    def save_visualization(self, image_path, output_path, img_size=(224, 224), alpha=0.4):
        """
        Generate and save Grad-CAM visualization
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            img_size: Image size for preprocessing
            alpha: Transparency factor
        """
        original_img, heatmap, superimposed_img, pred_class, confidence = \
            self.generate_visualization(image_path, img_size, alpha)
        
        if superimposed_img is None:
            print("Could not generate visualization")
            return
        
        # Convert RGB to BGR for saving
        superimposed_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, superimposed_bgr)
        print(f"Grad-CAM visualization saved to {output_path}")
        print(f"Prediction: {pred_class} (Confidence: {confidence:.2%})")


def create_gradcam(model_path='models/best_model.h5', image_path=None, output_path='gradcam_result.jpg'):
    """
    Convenience function to create Grad-CAM visualization
    
    Args:
        model_path: Path to model file
        image_path: Path to image file
        output_path: Path to save visualization
    """
    try:
        model = keras.models.load_model(model_path)
        model = ensure_model_built(model, input_shape=(224, 224, 3))
        gradcam = GradCAM(model, input_shape=(224, 224, 3))
        gradcam.save_visualization(image_path, output_path)
    except Exception as e:
        print(f"Error creating Grad-CAM: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Grad-CAM Visualization')
    parser.add_argument('image_path', type=str,
                       help='Path to image file')
    parser.add_argument('--model_path', type=str, default='models/best_model.h5',
                       help='Path to model file')
    parser.add_argument('--output_path', type=str, default='gradcam_result.jpg',
                       help='Path to save visualization')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Heatmap transparency (0-1)')
    
    args = parser.parse_args()
    
    try:
        model = keras.models.load_model(args.model_path)
        model = ensure_model_built(model, input_shape=(224, 224, 3))
        gradcam = GradCAM(model, input_shape=(224, 224, 3))
        gradcam.save_visualization(args.image_path, args.output_path, alpha=args.alpha)
    except Exception as e:
        print(f"Error: {e}")
