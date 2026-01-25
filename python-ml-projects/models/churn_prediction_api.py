
"""
Production Deployment Code for Churn Prediction
"""

import pandas as pd
import joblib

# Load artifacts
model = joblib.load('models/churn_predictor.pkl')
scaler = joblib.load('models/churn_scaler.pkl')
encoders = joblib.load('models/churn_encoders.pkl')
feature_columns = joblib.load('models/churn_features.pkl')

def predict_churn(customer_data):
    """
    Predict churn probability for a customer
    
    Args:
        customer_data: dict with customer features
        
    Returns:
        dict with prediction and probability
    """
    # Create DataFrame
    df = pd.DataFrame([customer_data])
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    # Ensure all features present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select and order features
    X = df[feature_columns]
    
    # Scale numerical features
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges',
                     'SupportTickets', 'LatePayments', 'ChargesPerMonth']
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    # Predict
    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability > 0.5)
    
    # Risk level
    if probability < 0.3:
        risk = 'Low'
    elif probability < 0.7:
        risk = 'Medium'
    else:
        risk = 'High'
    
    return {
        'will_churn': bool(prediction),
        'churn_probability': float(probability),
        'risk_level': risk,
        'confidence': float(max(probability, 1-probability))
    }

# Example usage:
if __name__ == '__main__':
    customer = {
        'Age': 45,
        'Gender': 'Male',
        'Tenure': 3,
        'MonthlyCharges': 85,
        'TotalCharges': 255,
        'InternetService': 'Fiber',
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic',
        'SupportTickets': 4,
        'LatePayments': 2,
        'ChargesPerMonth': 85,
        'IsNewCustomer': 1,
        'HighSupport': 1,
        'HasPaymentIssues': 1,
        'IsHighValue': 1
    }
    
    result = predict_churn(customer)
    print(f"Churn Prediction: {result}")
