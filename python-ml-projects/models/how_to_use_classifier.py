
# ============================================
# HOW TO USE SAVED MODEL FOR PREDICTIONS
# ============================================

from tensorflow import keras
import numpy as np
from PIL import Image
import json

# Load model and class names
model = keras.models.load_model('models/custom_shape_classifier.keras')
with open('models/custom_shape_classes.json', 'r') as f:
    class_names = json.load(f)

def predict_image(image_path):
    """Predict class of a single image"""
    # Load and preprocess
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return {
        'class': class_names[predicted_class],
        'confidence': float(confidence),
        'all_probabilities': {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
    }

# Example usage:
result = predict_image('new_image.png')
print(f"Predicted: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
