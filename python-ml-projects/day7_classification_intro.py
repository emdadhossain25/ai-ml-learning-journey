"""
Day 7: Classification Fundamentals
Understanding category prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("CLASSIFICATION: PREDICTING CATEGORIES")
print("=" * 60)

# ============================================
# CONCEPT 1: CLASSIFICATION VS REGRESSION
# ============================================

print("\n1. CLASSIFICATION vs REGRESSION")
print("-" * 60)

comparison = pd.DataFrame({
    'Aspect': ['Output Type', 'Example Task', 'Example Output', 'Metrics'],
    'Regression': [
        'Continuous number',
        'Predict house price',
        '$325,000',
        'RMSE, MAE, R²'
    ],
    'Classification': [
        'Discrete category',
        'Predict spam/not spam',
        'Spam',
        'Accuracy, Precision, Recall'
    ]
})

print(comparison.to_string(index=False))

# ============================================
# CONCEPT 2: BINARY CLASSIFICATION
# ============================================

print("\n" + "=" * 60)
print("2. BINARY CLASSIFICATION (2 classes)")
print("-" * 60)

print("Examples:")
print("  • Email: Spam (1) or Not Spam (0)")
print("  • Medical: Disease (1) or Healthy (0)")
print("  • Titanic: Survived (1) or Died (0)")
print("  • Finance: Fraud (1) or Legitimate (0)")

print("\nToday's focus: Binary Classification")

# ============================================
# CONCEPT 3: MULTI-CLASS CLASSIFICATION
# ============================================

print("\n" + "=" * 60)
print("3. MULTI-CLASS CLASSIFICATION (3+ classes)")
print("-" * 60)

print("Examples:")
print("  • Image: Cat (0), Dog (1), Bird (2)")
print("  • Iris: Setosa (0), Versicolor (1), Virginica (2)")
print("  • Handwriting: Digits 0-9")
print("  • Customer: Low (0), Medium (1), High (2) value")

# ============================================
# CONCEPT 4: PROBABILITY vs HARD PREDICTION
# ============================================

print("\n" + "=" * 60)
print("4. PROBABILITY PREDICTIONS")
print("-" * 60)

print("Classification models can output:")
print("\n1. HARD PREDICTION (class label):")
print("   'This email is SPAM'")

print("\n2. SOFT PREDICTION (probability):")
print("   'This email is 85% likely to be spam'")

print("\nExample predictions:")
examples = pd.DataFrame({
    'Email': ['Get rich quick!', 'Meeting at 3pm', 'You won lottery!', 'Project update'],
    'Spam_Probability': [0.95, 0.15, 0.88, 0.08],
    'Predicted_Class': ['Spam', 'Not Spam', 'Spam', 'Not Spam']
})
print(examples.to_string(index=False))

# ============================================
# CONCEPT 5: DECISION BOUNDARY
# ============================================

print("\n" + "=" * 60)
print("5. DECISION BOUNDARY (Threshold)")
print("-" * 60)

print("Default threshold: 0.5 (50%)")
print("  If probability >= 0.5 → Predict class 1")
print("  If probability < 0.5  → Predict class 0")

print("\nYou can adjust the threshold:")
print("  Strict (0.7): Only predict spam if >70% sure")
print("  Lenient (0.3): Predict spam if >30% sure")

# ============================================
# CONCEPT 5: DECISION BOUNDARY
# ============================================

print("\n" + "=" * 60)
print("5. DECISION BOUNDARY (Threshold)")
print("-" * 60)

print("Default threshold: 0.5 (50%)")
print("  If probability >= 0.5 → Predict class 1")
print("  If probability < 0.5  → Predict class 0")

print("\nYou can adjust the threshold:")
print("  Strict (0.7): Only predict spam if >70% sure")
print("  Lenient (0.3): Predict spam if >30% sure")

# ============================================
# CONCEPT 6: CLASSIFICATION EXAMPLE
# ============================================

print("\n" + "=" * 60)
print("6. SIMPLE CLASSIFICATION EXAMPLE")
print("-" * 60)

# Simple dataset: Study hours → Pass/Fail
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
passed = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])  # 0=Failed, 1=Passed

print("Student exam results:")
for hours, result in zip(study_hours, passed):
    status = "Passed" if result == 1 else "Failed"
    print(f"  {hours} hours study → {status}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Raw data
axes[0].scatter(study_hours, passed, c=passed, cmap='RdYlGn', 
               s=100, edgecolors='black', linewidth=2)
axes[0].set_xlabel('Study Hours', fontsize=12)
axes[0].set_ylabel('Result (0=Fail, 1=Pass)', fontsize=12)
axes[0].set_title('Classification Problem', fontsize=14, fontweight='bold')
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['Failed', 'Passed'])
axes[0].grid(alpha=0.3)

# Plot 2: With decision boundary
axes[1].scatter(study_hours, passed, c=passed, cmap='RdYlGn',
               s=100, edgecolors='black', linewidth=2)
axes[1].axvline(x=4.5, color='blue', linestyle='--', linewidth=3,
               label='Decision Boundary')
axes[1].fill_betweenx([0, 1], 0, 4.5, alpha=0.2, color='red', label='Predict: Fail')
axes[1].fill_betweenx([0, 1], 4.5, 10, alpha=0.2, color='green', label='Predict: Pass')
axes[1].set_xlabel('Study Hours', fontsize=12)
axes[1].set_ylabel('Result', fontsize=12)
axes[1].set_title('With Decision Boundary', fontsize=14, fontweight='bold')
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Failed', 'Passed'])
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/27_classification_intro.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: plots/27_classification_intro.png")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("1. Classification predicts categories, not numbers")
print("2. Binary classification: 2 classes (yes/no, 0/1)")
print("3. Models output probabilities (0-1 range)")
print("4. Threshold (usually 0.5) converts probability to class")
print("5. Decision boundary separates classes")

print("\n✅ Classification concepts complete!")
print("Next: Build your first classifier!")
























































