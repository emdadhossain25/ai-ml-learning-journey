"""
Day 6: scikit-learn Fundamentals
Understanding the ML library that powers real-world AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

print("=" * 60)
print("SCIKIT-LEARN FUNDAMENTALS")
print("=" * 60)

# ============================================
# 1. CREATE REALISTIC DATASET
# ============================================

print("\n1. Creating realistic dataset...")
print("-" * 60)

# House prices dataset
np.random.seed(42)
n_samples = 100

# Features
size = np.random.randint(1000, 3500, n_samples)
bedrooms = np.random.randint(2, 6, n_samples)
age = np.random.randint(0, 30, n_samples)
distance_to_city = np.random.uniform(1, 20, n_samples)

# Target: Price (with realistic relationship)
price = (200 * size +  # $200 per sqft
         50000 * bedrooms +  # $50k per bedroom
         -2000 * age +  # Depreciation
         -5000 * distance_to_city +  # Location matters
         np.random.randn(n_samples) * 50000)  # Noise

# Create DataFrame
df = pd.DataFrame({
    'size_sqft': size,
    'bedrooms': bedrooms,
    'age_years': age,
    'distance_km': distance_to_city,
    'price': price
})

print(df.head(10))
print(f"\nDataset: {len(df)} houses with 4 features")

# ============================================
# 2. DATA PREPARATION
# ============================================

print("\n2. Preparing data...")
print("-" * 60)

# Separate features and target
X = df[['size_sqft', 'bedrooms', 'age_years', 'distance_km']]
y = df['price']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================
# 3. FEATURE SCALING (IMPORTANT!)
# ============================================

print("\n3. Feature scaling...")
print("-" * 60)

print("Before scaling:")
print(X_train.describe())

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nAfter scaling (first 3 samples):")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head(3))
print("\n✅ All features now on similar scale!")

# ============================================
# 4. TRAIN MODEL
# ============================================

print("\n4. Training model...")
print("-" * 60)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✅ Model trained!")
print("\nLearned coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:,.0f}")
print(f"  Intercept: {model.intercept_:,.0f}")

# ============================================
# 5. MAKE PREDICTIONS
# ============================================

print("\n5. Making predictions...")
print("-" * 60)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Sample predictions (first 5 test samples):")
comparison = pd.DataFrame({
    'Actual': y_test[:5].values,
    'Predicted': y_test_pred[:5],
    'Difference': y_test[:5].values - y_test_pred[:5]
})
print(comparison)

# ============================================
# 6. COMPREHENSIVE EVALUATION
# ============================================

print("\n6. Model evaluation...")
print("-" * 60)

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("TRAINING SET PERFORMANCE:")
print(f"  RMSE: ${train_rmse:,.0f}")
print(f"  MAE:  ${train_mae:,.0f}")
print(f"  R²:   {train_r2:.3f}")

print("\nTEST SET PERFORMANCE:")
print(f"  RMSE: ${test_rmse:,.0f}")
print(f"  MAE:  ${test_mae:,.0f}")
print(f"  R²:   {test_r2:.3f}")

print("\nWhat do these metrics mean?")
print(f"  RMSE: Predictions are off by ${test_rmse:,.0f} on average")
print(f"  MAE:  Average absolute error is ${test_mae:,.0f}")
print(f"  R²:   Model explains {test_r2*100:.1f}% of price variance")

# Check for overfitting
print("\nOverfitting check:")
if abs(train_r2 - test_r2) < 0.05:
    print("  ✅ Model generalizes well (not overfitting)")
else:
    print("  ⚠️  Possible overfitting (train score much higher than test)")

# ============================================
# 7. CROSS-VALIDATION
# ============================================

print("\n7. Cross-validation (more robust evaluation)...")
print("-" * 60)

# 5-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                            cv=5, scoring='r2')

print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f}")
print(f"Std deviation: {cv_scores.std():.3f}")
print("\n✅ Cross-validation gives more reliable performance estimate")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================

print("\n8. Feature importance...")
print("-" * 60)

# Absolute coefficients show importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(importance)
print("\nMost important features:")
print(importance.head(2)[['Feature', 'Coefficient']])

# ============================================
# VISUALIZATIONS
# ============================================

print("\n9. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, s=50)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect prediction')
axes[0, 0].set_xlabel('Actual Price ($)', fontsize=12)
axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=12)
axes[0, 0].set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residuals (errors)
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, s=50)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12)
axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Residual distribution
axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residual ($)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Feature importance
axes[1, 1].barh(importance['Feature'], importance['Abs_Coefficient'])
axes[1, 1].set_xlabel('Absolute Coefficient', fontsize=12)
axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/25_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/25_model_evaluation.png")

# ============================================
# 10. MAKE PREDICTIONS FOR NEW HOUSES
# ============================================

print("\n10. Predicting prices for NEW houses...")
print("-" * 60)

new_houses = pd.DataFrame({
    'size_sqft': [2000, 2500, 1500],
    'bedrooms': [3, 4, 2],
    'age_years': [5, 10, 20],
    'distance_km': [5, 10, 3]
})

print("New houses:")
print(new_houses)

# Scale features
new_houses_scaled = scaler.transform(new_houses)

# Predict
predictions = model.predict(new_houses_scaled)

print("\nPredicted prices:")
for i, price in enumerate(predictions):
    print(f"  House {i+1}: ${price:,.0f}")

print("\n" + "=" * 60)
print("✅ SCIKIT-LEARN MASTERY COMPLETE!")
print("=" * 60)
print("\nYou now know:")
print("  • Data preparation and splitting")
print("  • Feature scaling")
print("  • Model training and prediction")
print("  • Comprehensive evaluation (RMSE, MAE, R²)")
print("  • Cross-validation")
print("  • Feature importance")
print("  • Making predictions for new data")