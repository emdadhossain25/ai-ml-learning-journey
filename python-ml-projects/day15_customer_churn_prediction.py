"""
Day 15: Customer Churn Prediction - Portfolio Project
Complete ML system for predicting customer churn
A production-ready project demonstrating end-to-end ML skills
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CUSTOMER CHURN PREDICTION - PORTFOLIO PROJECT")
print("=" * 70)

# ============================================
# BUSINESS PROBLEM
# ============================================

print("\n" + "=" * 70)
print("1. BUSINESS PROBLEM DEFINITION")
print("=" * 70)

print("""
BUSINESS CONTEXT:
  • Telecom company losing customers (churn)
  • Acquiring new customer costs 5-10× more than retention
  • Need to predict which customers will leave
  • Goal: Proactive retention campaigns

SUCCESS METRICS:
  • Predict churn with 80%+ accuracy
  • Identify high-risk customers
  • Provide actionable insights
  • Deploy as production system

VALUE PROPOSITION:
  • If 1,000 customers churn/month
  • Each worth $50/month
  • Loss: $50,000/month
  • Preventing 50% saves $25,000/month!

MY ROLE: Build predictive model + deployment system
""")

# ============================================
# CREATE SYNTHETIC DATASET
# ============================================

print("\n" + "=" * 70)
print("2. CREATING REALISTIC CUSTOMER DATA")
print("=" * 70)

print("Generating synthetic telecom customer dataset...")

np.random.seed(42)
n_customers = 5000

# Customer demographics
customer_id = [f"CUST{i:05d}" for i in range(1, n_customers + 1)]
age = np.random.normal(45, 15, n_customers).clip(18, 80).astype(int)
gender = np.random.choice(['Male', 'Female'], n_customers)

# Account information
tenure = np.random.exponential(24, n_customers).clip(0, 72).astype(int)
monthly_charges = np.random.normal(65, 25, n_customers).clip(20, 150)
total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_customers)

# Services
internet_service = np.random.choice(['DSL', 'Fiber', 'No'], n_customers, 
                                   p=[0.4, 0.4, 0.2])
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                            n_customers, p=[0.5, 0.3, 0.2])
payment_method = np.random.choice(['Electronic', 'Mailed check', 'Bank transfer', 'Credit card'],
                                  n_customers, p=[0.3, 0.25, 0.25, 0.2])

# Support interactions
support_tickets = np.random.poisson(2, n_customers)
late_payments = np.random.poisson(1, n_customers)

# Churn (target variable) - influenced by features
churn_probability = (
    0.1 +  # Base churn rate
    0.3 * (contract == 'Month-to-month') +  # Higher for monthly
    0.2 * (tenure < 6) +  # New customers churn more
    0.15 * (monthly_charges > 80) +  # High bills → churn
    0.1 * (support_tickets > 3) +  # Many complaints → churn
    0.1 * (late_payments > 2)  # Payment issues → churn
)

churn_probability = churn_probability.clip(0, 1)
churn = (np.random.random(n_customers) < churn_probability).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': customer_id,
    'Age': age,
    'Gender': gender,
    'Tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService': internet_service,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'SupportTickets': support_tickets,
    'LatePayments': late_payments,
    'Churn': churn
})

print(f"\n✅ Dataset created:")
print(f"   Customers: {len(df):,}")
print(f"   Features: {len(df.columns) - 2}")  # Excluding CustomerID and Churn
print(f"   Churn rate: {churn.mean():.1%}")

print(f"\nFirst 5 customers:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

# ============================================
# EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nChurn Statistics:")
churn_stats = df.groupby('Churn').agg({
    'CustomerID': 'count',
    'Tenure': 'mean',
    'MonthlyCharges': 'mean',
    'SupportTickets': 'mean'
}).round(2)
churn_stats.columns = ['Count', 'Avg Tenure', 'Avg Monthly Charges', 'Avg Support Tickets']
print(churn_stats)

print("\nKey Insights:")
churned = df[df['Churn'] == 1]
retained = df[df['Churn'] == 0]

print(f"  • Churned customers have {churned['Tenure'].mean():.1f} months tenure")
print(f"  • Retained customers have {retained['Tenure'].mean():.1f} months tenure")
print(f"  • Churned pay ${churned['MonthlyCharges'].mean():.2f}/month on average")
print(f"  • Retained pay ${retained['MonthlyCharges'].mean():.2f}/month on average")

# Churn by contract type
print(f"\nChurn Rate by Contract Type:")
contract_churn = df.groupby('Contract')['Churn'].agg(['sum', 'count', 'mean'])
contract_churn['ChurnRate'] = (contract_churn['mean'] * 100).round(1)
print(contract_churn[['ChurnRate']])

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n" + "=" * 70)
print("4. FEATURE ENGINEERING")
print("=" * 70)

# Create new features
df['ChargesPerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)  # Avoid division by zero
df['IsNewCustomer'] = (df['Tenure'] < 6).astype(int)
df['HighSupport'] = (df['SupportTickets'] > 3).astype(int)
df['HasPaymentIssues'] = (df['LatePayments'] > 1).astype(int)
df['IsHighValue'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

print("✅ New features created:")
print("   • ChargesPerMonth: Average monthly cost")
print("   • IsNewCustomer: Tenure < 6 months")
print("   • HighSupport: More than 3 support tickets")
print("   • HasPaymentIssues: More than 1 late payment")
print("   • IsHighValue: Above median monthly charges")

# ============================================
# PREPARE DATA
# ============================================

print("\n" + "=" * 70)
print("5. PREPARING DATA FOR MODELING")
print("=" * 70)

# Select features
feature_columns = [
    'Age', 'Tenure', 'MonthlyCharges', 'TotalCharges',
    'SupportTickets', 'LatePayments', 'ChargesPerMonth',
    'IsNewCustomer', 'HighSupport', 'HasPaymentIssues', 'IsHighValue',
    'Gender', 'InternetService', 'Contract', 'PaymentMethod'
]

X = df[feature_columns].copy()
y = df['Churn'].values

# Encode categorical variables
categorical_features = ['Gender', 'InternetService', 'Contract', 'PaymentMethod']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✅ Data prepared:")
print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]:,}")
print(f"   Churn rate: {y.mean():.1%}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                     'SupportTickets', 'LatePayments', 'ChargesPerMonth']

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"\n✅ Train-test split:")
print(f"   Training: {len(X_train):,} customers")
print(f"   Test: {len(X_test):,} customers")

# ============================================
# MODEL TRAINING & COMPARISON
# ============================================

print("\n" + "=" * 70)
print("6. TRAINING MULTIPLE MODELS")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Results summary
results_df = pd.DataFrame(results)
print(f"\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name}")

# Use best model
best_model = models[best_model_name]

# ============================================
# DETAILED EVALUATION
# ============================================

print("\n" + "=" * 70)
print("7. DETAILED MODEL EVALUATION")
print("=" * 70)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Retained  Churned")
print(f"Actual Retained    {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"       Churned     {cm[1,0]:4d}     {cm[1,1]:4d}")

# Business metrics
true_negatives = cm[0, 0]
false_positives = cm[0, 1]
false_negatives = cm[1, 0]
true_positives = cm[1, 1]

print(f"\nBusiness Impact:")
print(f"  • Correctly identified churners: {true_positives}")
print(f"  • Missed churners: {false_negatives}")
print(f"  • False alarms: {false_positives}")
print(f"  • Precision (when we predict churn, we're right): {true_positives/(true_positives+false_positives):.1%}")
print(f"  • Recall (we catch this % of churners): {true_positives/(true_positives+false_negatives):.1%}")

# ============================================
# FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 70)
print("8. FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\nKey Drivers of Churn:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  • {row['Feature']}: {row['Importance']:.4f}")

# ============================================
# RISK SCORING
# ============================================

print("\n" + "=" * 70)
print("9. CUSTOMER RISK SCORING")
print("=" * 70)

# Add risk scores to test set
test_df = df.iloc[X_test.index].copy()
test_df['ChurnProbability'] = y_pred_proba
test_df['RiskLevel'] = pd.cut(y_pred_proba, 
                              bins=[0, 0.3, 0.7, 1.0],
                              labels=['Low', 'Medium', 'High'])

print("Risk Level Distribution:")
print(test_df['RiskLevel'].value_counts())

print("\nHigh-Risk Customers (Sample):")
high_risk = test_df[test_df['RiskLevel'] == 'High'].nlargest(5, 'ChurnProbability')
print(high_risk[['CustomerID', 'Tenure', 'MonthlyCharges', 'Contract', 'ChurnProbability']].to_string(index=False))

# ============================================
# SAVE MODEL & ARTIFACTS
# ============================================

print("\n" + "=" * 70)
print("10. SAVING PRODUCTION ARTIFACTS")
print("=" * 70)

# Save model
joblib.dump(best_model, 'models/churn_predictor.pkl')
print("✅ Model saved: models/churn_predictor.pkl")

# Save scaler
joblib.dump(scaler, 'models/churn_scaler.pkl')
print("✅ Scaler saved: models/churn_scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'models/churn_encoders.pkl')
print("✅ Encoders saved: models/churn_encoders.pkl")

# Save feature names
joblib.dump(feature_columns, 'models/churn_features.pkl')
print("✅ Features saved: models/churn_features.pkl")

# ============================================
# DEPLOYMENT CODE
# ============================================

print("\n" + "=" * 70)
print("11. GENERATING DEPLOYMENT CODE")
print("=" * 70)

deployment_code = '''
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
'''

with open('models/churn_prediction_api.py', 'w') as f:
    f.write(deployment_code)

print("✅ Deployment code saved: models/churn_prediction_api.py")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 70)
print("12. CREATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)

fig.suptitle('Customer Churn Prediction - Portfolio Project', 
             fontsize=20, fontweight='bold')

# Plot 1: Churn Distribution
ax1 = fig.add_subplot(gs[0, 0])
churn_counts = df['Churn'].value_counts()
ax1.bar(['Retained', 'Churned'], churn_counts.values, 
       color=['lightgreen', 'lightcoral'], edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
ax1.set_title('Churn Distribution', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(churn_counts.values):
    ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')

# Plot 2: Churn by Contract Type
ax2 = fig.add_subplot(gs[0, 1])
contract_data = df.groupby('Contract')['Churn'].mean() * 100
contract_data.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black', linewidth=2)
ax2.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Churn Rate by Contract Type', fontsize=13, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Tenure Distribution
ax3 = fig.add_subplot(gs[0, 2])
churned['Tenure'].hist(bins=20, alpha=0.6, label='Churned', color='red', ax=ax3)
retained['Tenure'].hist(bins=20, alpha=0.6, label='Retained', color='green', ax=ax3)
ax3.set_xlabel('Tenure (months)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
ax3.set_title('Tenure Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Model Comparison
ax4 = fig.add_subplot(gs[1, :])
x_pos = np.arange(len(results_df))
width = 0.35
ax4.bar(x_pos - width/2, results_df['Accuracy'], width, 
       label='Accuracy', color='lightblue', edgecolor='black')
ax4.bar(x_pos + width/2, results_df['ROC-AUC'], width,
       label='ROC-AUC', color='lightcoral', edgecolor='black')
ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(results_df['Model'])
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Confusion Matrix
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=['Retained', 'Churned'],
           yticklabels=['Retained', 'Churned'],
           annot_kws={'fontsize': 14})
ax5.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax5.set_title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')

# Plot 6: ROC Curve
ax6 = fig.add_subplot(gs[2, 1])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)
ax6.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {auc_score:.3f})', color='blue')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax6.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax6.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax6.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

# Plot 7: Feature Importance
ax7 = fig.add_subplot(gs[2, 2])
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(10)
    ax7.barh(range(len(top_features)), top_features['Importance'],
            color='lightgreen', edgecolor='black')
    ax7.set_yticks(range(len(top_features)))
    ax7.set_yticklabels(top_features['Feature'], fontsize=9)
    ax7.set_xlabel('Importance', fontsize=11, fontweight='bold')
    ax7.set_title('Top 10 Features', fontsize=13, fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(axis='x', alpha=0.3)

# Plot 8: Risk Distribution
ax8 = fig.add_subplot(gs[3, :])
ax8.hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax8.axvline(x=0.3, color='green', linestyle='--', linewidth=2, label='Low Risk Threshold')
ax8.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
ax8.set_xlabel('Churn Probability', fontsize=12, fontweight='bold')
ax8.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax8.set_title('Customer Risk Score Distribution', fontsize=14, fontweight='bold')
ax8.legend(fontsize=11)
ax8.grid(axis='y', alpha=0.3)

plt.savefig('plots/57_churn_prediction_portfolio.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/57_churn_prediction_portfolio.png")

# ============================================
# PROJECT SUMMARY
# ============================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY - CUSTOMER CHURN PREDICTION")
print("=" * 70)

print(f"""
BUSINESS PROBLEM:
  • Telecom company losing {df['Churn'].sum():,} customers
  • Each customer worth ${df['MonthlyCharges'].mean():.2f}/month
  • Annual revenue at risk: ${df['Churn'].sum() * df['MonthlyCharges'].mean() * 12:,.0f}

SOLUTION DELIVERED:
  • Predictive model with {results_df['ROC-AUC'].max():.1%} ROC-AUC
  • Identifies {true_positives} out of {true_positives + false_negatives} churners
  • Enables proactive retention campaigns
  • Production-ready deployment code

MODEL PERFORMANCE:
  • Algorithm: {best_model_name}
  • Accuracy: {results_df.loc[results_df['Model'] == best_model_name, 'Accuracy'].values[0]:.2%}
  • ROC-AUC: {results_df.loc[results_df['Model'] == best_model_name, 'ROC-AUC'].values[0]:.2%}
  • Precision: {true_positives/(true_positives+false_positives):.1%}
  • Recall: {true_positives/(true_positives+false_negatives):.1%}

KEY INSIGHTS:
  • Contract type is strongest predictor
  • New customers (< 6 months) churn most
  • High support tickets indicate dissatisfaction
  • Month-to-month contracts have {contract_churn.loc['Month-to-month', 'ChurnRate']:.1f}% churn rate

DELIVERABLES:
  ✅ Trained ML model
  ✅ Feature engineering pipeline
  ✅ Production deployment code
  ✅ Customer risk scoring system
  ✅ Actionable insights for business

BUSINESS IMPACT:
  If model prevents 50% of churn:
  • Customers saved: {int((true_positives + false_negatives) * 0.5):,}
  • Monthly revenue saved: ${int((true_positives + false_negatives) * 0.5 * df['MonthlyCharges'].mean()):,}
  • Annual impact: ${int((true_positives + false_negatives) * 0.5 * df['MonthlyCharges'].mean() * 12):,}

NEXT STEPS:
  1. A/B test retention campaigns on high-risk customers
  2. Monitor model performance monthly
  3. Retrain with new data quarterly
  4. Deploy as real-time API for customer service

THIS PROJECT DEMONSTRATES:
  ✓ End-to-end ML workflow
  ✓ Business problem understanding
  ✓ Data analysis & visualization
  ✓ Feature engineering
  ✓ Model selection & tuning
  ✓ Production deployment
  ✓ Business impact quantification

PORTFOLIO READY! 🎯
""")

print("\n✅ Customer Churn Prediction project complete!")
print("\n🎉 This is a STRONG portfolio piece! 🎉")