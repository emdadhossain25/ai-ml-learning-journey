"""
Day 6: Building Your First Machine Learning Model
Linear Regression - The foundation of ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 60)
print("YOUR FIRST MACHINE LEARNING MODEL!")
print("=" * 60)

# ============================================
# STEP 1: CREATE DATASET
# ============================================

print("\nSTEP 1: Creating dataset...")
print("-" * 60)

# Study hours vs Exam scores
np.random.seed(42)
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# True relationship: score = 10 * hours + 40, with some noise
exam_scores = 10 * study_hours.flatten() + 40 + np.random.randn(10) * 3

print("Our dataset:")
for hours, score in zip(study_hours.flatten(), exam_scores):
    print(f"  {hours} hours → {score:.1f} points")

# ============================================
# STEP 2: SPLIT DATA
# ============================================

print("\nSTEP 2: Splitting into train and test...")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    study_hours, exam_scores, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================
# STEP 3: CREATE MODEL
# ============================================

print("\nSTEP 3: Creating the model...")
print("-" * 60)

model = LinearRegression()
print("✅ Model created: Linear Regression")
print("   Formula: y = mx + b")
print("   Model will learn 'm' (slope) and 'b' (intercept)")

# ============================================
# STEP 4: TRAIN MODEL
# ============================================

print("\nSTEP 4: Training the model...")
print("-" * 60)

model.fit(X_train, y_train)

print("✅ Model trained!")
print(f"   Learned slope (m): {model.coef_[0]:.2f}")
print(f"   Learned intercept (b): {model.intercept_:.2f}")
print(f"   Formula: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

# ============================================
# STEP 5: MAKE PREDICTIONS
# ============================================

print("\nSTEP 5: Making predictions...")
print("-" * 60)

# Predict on test set
y_pred = model.predict(X_test)

print("Test set predictions:")
for hours, actual, predicted in zip(X_test.flatten(), y_test, y_pred):
    print(f"  {hours} hours → Actual: {actual:.1f}, Predicted: {predicted:.1f}")

# ============================================
# STEP 6: EVALUATE MODEL
# ============================================

print("\nSTEP 6: Evaluating model performance...")
print("-" * 60)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

print("\nWhat do these mean?")
print(f"  RMSE: On average, predictions are off by {rmse:.1f} points")
print(f"  R² Score: Model explains {r2*100:.1f}% of the variance")
print(f"           (1.0 = perfect, 0.0 = useless)")

# ============================================
# STEP 7: USE MODEL FOR NEW PREDICTIONS
# ============================================

print("\nSTEP 7: Using model for NEW predictions...")
print("-" * 60)

# Predict for students who studied 3.5 and 7.5 hours
new_hours = np.array([[3.5], [7.5]])
new_predictions = model.predict(new_hours)

print("Predictions for new data:")
for hours, pred in zip(new_hours.flatten(), new_predictions):
    print(f"  {hours} hours → Predicted score: {pred:.1f}")

# ============================================
# VISUALIZATION
# ============================================

print("\nCreating visualization...")

plt.figure(figsize=(12, 5))

# Plot 1: Training data and model
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=100, label='Training data')
plt.plot(study_hours, model.predict(study_hours), 'r-', linewidth=3, label='Learned model')
plt.title('Model Training', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Test predictions
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', alpha=0.6, s=100, label='Actual test scores')
plt.scatter(X_test, y_pred, color='red', alpha=0.6, s=100, label='Predicted scores')
plt.plot(study_hours, model.predict(study_hours), 'r--', linewidth=2, alpha=0.5)
plt.title('Model Predictions on Test Set', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/24_first_ml_model.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/24_first_ml_model.png")

print("\n" + "=" * 60)
print("🎉 CONGRATULATIONS! 🎉")
print("=" * 60)
print("You just built, trained, and evaluated your first ML model!")
print("You are now officially a Machine Learning practitioner!")