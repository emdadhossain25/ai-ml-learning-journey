"""
Day 10: Flask REST API for ML Model
Deploy your model as a web service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained pipeline
MODEL_PATH = 'models/titanic_pipeline.pkl'

print("=" * 60)
print("LOADING ML MODEL")
print("=" * 60)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Make sure you ran day10_ml_pipelines.py first!")
    model = None

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Titanic Survival Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'API documentation',
            '/health': 'Health check',
            '/predict': 'Make prediction (POST)',
            '/predict_batch': 'Batch predictions (POST)'
        },
        'example_request': {
            'Pclass': 1,
            'Name': 'Miss. Jane Smith',
            'Sex': 'female',
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 100,
            'Embarked': 'C',
            'Cabin': 'C85',
            'Ticket': '12345'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    status = 'healthy' if model is not None else 'unhealthy'
    return jsonify({
        'status': status,
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict survival for a single passenger
    
    Expects JSON with passenger data:
    {
        "Pclass": 1,
        "Name": "Miss. Jane Smith",
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C",
        "Cabin": "C85",
        "Ticket": "12345"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        passenger_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(passenger_df)[0]
        probability = model.predict_proba(passenger_df)[0, 1]
        
        # Prepare response
        result = {
            'survived': bool(prediction),
            'survival_probability': float(probability),
            'prediction': 'Survived' if prediction == 1 else 'Did not survive',
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.15 else 'Low'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict survival for multiple passengers
    
    Expects JSON array of passenger data
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        passengers_df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(passengers_df)
        probabilities = model.predict_proba(passengers_df)[:, 1]
        
        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'passenger_id': i,
                'survived': bool(pred),
                'survival_probability': float(prob),
                'prediction': 'Survived' if pred == 1 else 'Did not survive'
            })
        
        return jsonify({'predictions': results, 'count': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STARTING FLASK API SERVER")
    print("=" * 60)
    print("\nAPI will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /         - API documentation")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Single prediction")
    print("  POST /predict_batch - Batch predictions")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)