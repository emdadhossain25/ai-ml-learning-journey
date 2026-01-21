"""
Day 7: Logistic Regression
Your first classification model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("LOGISTIC REGRESSION - FIRST CLASSIFIER")
print("=" * 60)

# ============================================
# STEP 1: CREATE DATASET
# ============================================

print("\nSTEP 1: Creating dataset...")
print("-" * 60)

# Student performance: study_hours + sleep_hours → Pass/Fail
np.random.seed(42)
n_samples = 100

study_hours = np.random.uniform(1, 10, n_samples)
sleep_hours = np.random.uniform(4, 10, n_samples)

# Students who study more AND sleep well are more likely to pass
# Create realistic pass/fail based on both factors
score = 2 * study_hours + 1.5 * sleep_hours + np.random.randn(n_samples) * 2
passed = (score > 20).astype(int)  # Threshold to create binary outcome

df = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'passed': passed
})

print(f"Created {n_samples} student records")
print("\nFirst few students:")
print(df.head(10))

print(f"\nClass distribution:")
print(df['passed'].value_counts())
print(f"  Pass rate: {df['passed'].mean()*100:.1f}%")

# ============================================
# STEP 2: VISUALIZE DATA
# ============================================

print("\nSTEP 2: Visualizing the data...")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['study_hours'], df['sleep_hours'], 
                     c=df['passed'], cmap='RdYlGn', 
                     s=100, edgecolors='black', linewidth=1.5, alpha=0.7)
plt.colorbar(scatter, label='Passed (1) / Failed (0)')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Sleep Hours', fontsize=12)
plt.title('Student Performance Dataset', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('plots/28_student_data.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/28_student_data.png")

# ============================================
# STEP 3: PREPARE DATA
# ============================================

print("\nSTEP 3: Preparing data...")
print("-" * 60)

X = df[['study_hours', 'sleep_hours']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled")

# ============================================
# STEP 4: TRAIN LOGISTIC REGRESSION
# ============================================

print("\nSTEP 4: Training Logistic Regression...")
print("-" * 60)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Model trained!")
print(f"\nLearned coefficients:")
print(f"  Study hours: {model.coef_[0][0]:.3f}")
print(f"  Sleep hours: {model.coef_[0][1]:.3f}")
print(f"  Intercept: {model.intercept_[0]:.3f}")

# ============================================
# STEP 5: MAKE PREDICTIONS
# ============================================

print("\nSTEP 5: Making predictions...")
print("-" * 60)

# Get probability predictions
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of passing

# Get class predictions
y_pred = model.predict(X_test_scaled)

print("Sample predictions (first 10 test samples):")
results = pd.DataFrame({
    'Study_Hours': X_test['study_hours'].values[:10],
    'Sleep_Hours': X_test['sleep_hours'].values[:10],
    'Actual': y_test.values[:10],
    'Prob_Pass': y_pred_proba[:10],
    'Predicted': y_pred[:10]
})
results['Actual'] = results['Actual'].map({0: 'Fail', 1: 'Pass'})
results['Predicted'] = results['Predicted'].map({0: 'Fail', 1: 'Pass'})
print(results.to_string(index=False))

# ============================================
# STEP 6: EVALUATE MODEL
# ============================================

print("\n" + "=" * 60)
print("STEP 6: MODEL EVALUATION")
print("=" * 60)

# Accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy:")
print(f"  Training: {train_accuracy*100:.2f}%")
print(f"  Test: {test_accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
print("\nInterpretation:")
print(f"  True Negatives (correctly predicted Fail): {cm[0,0]}")
print(f"  False Positives (predicted Pass, actually Fail): {cm[0,1]}")
print(f"  False Negatives (predicted Fail, actually Pass): {cm[1,0]}")
print(f"  True Positives (correctly predicted Pass): {cm[1,1]}")

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['Failed', 'Passed']))

# ============================================
# STEP 7: VISUALIZE RESULTS
# ============================================

print("\nSTEP 7: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Decision boundary
ax1 = axes[0, 0]
h = 0.02  # Step size in the mesh
x_min, x_max = X['study_hours'].min() - 1, X['study_hours'].max() + 1
y_min, y_max = X['sleep_hours'].min() - 1, X['sleep_hours'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Scale the mesh grid
mesh_data = np.c_[xx.ravel(), yy.ravel()]
mesh_data_scaled = scaler.transform(mesh_data)
Z = model.predict(mesh_data_scaled)
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
ax1.scatter(X_test['study_hours'], X_test['sleep_hours'], 
           c=y_test, cmap='RdYlGn', s=100, edgecolors='black', linewidth=1.5)
ax1.set_xlabel('Study Hours', fontsize=12)
ax1.set_ylabel('Sleep Hours', fontsize=12)
ax1.set_title('Decision Boundary', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# Plot 2: Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
           xticklabels=['Failed', 'Passed'],
           yticklabels=['Failed', 'Passed'])
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 3: Probability distribution
ax3 = axes[1, 0]
failed_probs = y_pred_proba[y_test == 0]
passed_probs = y_pred_proba[y_test == 1]

ax3.hist(failed_probs, bins=20, alpha=0.6, label='Actually Failed', 
        color='red', edgecolor='black')
ax3.hist(passed_probs, bins=20, alpha=0.6, label='Actually Passed',
        color='green', edgecolor='black')
ax3.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, 
           label='Decision Threshold')
ax3.set_xlabel('Predicted Probability of Passing', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Probability Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Predictions vs Actual
ax4 = axes[1, 1]
correct = y_test == y_pred
ax4.scatter(X_test['study_hours'][correct], X_test['sleep_hours'][correct],
           c='green', s=100, label='Correct', alpha=0.7, edgecolors='black')
ax4.scatter(X_test['study_hours'][~correct], X_test['sleep_hours'][~correct],
           c='red', s=100, marker='x',  label='Incorrect', linewidth=3)
ax4.set_xlabel('Study Hours', fontsize=12)
ax4.set_ylabel('Sleep Hours', fontsize=12)
ax4.set_title('Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/29_logistic_regression_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/29_logistic_regression_results.png")

# ============================================
# STEP 8: PREDICT FOR NEW STUDENTS
# ============================================

print("\n" + "=" * 60)
print("STEP 8: Predicting for NEW students")
print("=" * 60)

new_students = pd.DataFrame({
    'study_hours': [3, 7, 5, 9],
    'sleep_hours': [5, 8, 6, 7]
})

print("\nNew students:")
print(new_students)

new_students_scaled = scaler.transform(new_students)
new_predictions = model.predict(new_students_scaled)
new_probabilities = model.predict_proba(new_students_scaled)[:, 1]

print("\nPredictions:")
for i, (pred, prob) in enumerate(zip(new_predictions, new_probabilities)):
    result = "PASS" if pred == 1 else "FAIL"
    print(f"  Student {i+1}: {result} (probability: {prob:.2%})")

print("\n" + "=" * 60)
print("🎉 FIRST CLASSIFIER COMPLETE!")
print("=" * 60)
print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
print("You just built a working classification model!")