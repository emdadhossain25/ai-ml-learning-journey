"""
Day 6: Machine Learning Concepts
Understanding the fundamentals before building models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("MACHINE LEARNING CONCEPTS")
print("=" * 60)

# ============================================
# CONCEPT 1: SUPERVISED LEARNING
# ============================================

print("\n1. SUPERVISED LEARNING")
print("-" * 60)
print("We have INPUT (X) and OUTPUT (y)")
print("Goal: Learn the relationship between X and y")
print("Then use it to predict y for new X values")

# Example: Study hours vs Exam scores
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
exam_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85])

print("\nExample Dataset:")
for hours, score in zip(study_hours, exam_scores):
    print(f"  {hours} hours study → {score} points")

print("\nPattern: More study hours → Higher scores")
print("ML will learn this relationship automatically!")

# ============================================
# CONCEPT 2: FEATURES (X) AND TARGET (y)
# ============================================

print("\n" + "=" * 60)
print("2. FEATURES (X) AND TARGET (y)")
print("-" * 60)

print("FEATURES (X): Input variables we use to make predictions")
print("  Example: house size, bedrooms, location")
print("\nTARGET (y): What we want to predict")
print("  Example: house price")

# Example dataset
data = {
    'size_sqft': [1500, 1800, 2200, 1600, 2500],
    'bedrooms': [3, 3, 4, 3, 4],
    'age_years': [10, 5, 15, 8, 2],
    'price': [300000, 350000, 400000, 320000, 450000]
}
df = pd.DataFrame(data)

print("\nHouse Price Dataset:")
print(df)

print("\nFeatures (X):", ['size_sqft', 'bedrooms', 'age_years'])
print("Target (y):", ['price'])

# ============================================
# CONCEPT 3: TRAINING AND TESTING
# ============================================

print("\n" + "=" * 60)
print("3. TRAINING AND TESTING")
print("-" * 60)

print("TRAINING SET: Data we use to teach the model")
print("  Model learns patterns from this data")
print("\nTEST SET: Data we use to evaluate the model")
print("  Model has never seen this before")
print("  Tells us if the model really learned or just memorized")

print("\nTypical split: 80% training, 20% testing")

# Visualize the split
total_data = 100
train_size = 80
test_size = 20

print(f"\nExample: {total_data} houses total")
print(f"  Train on: {train_size} houses")
print(f"  Test on: {test_size} houses")

# ============================================
# CONCEPT 4: MODEL TYPES
# ============================================

print("\n" + "=" * 60)
print("4. TYPES OF ML PROBLEMS")
print("-" * 60)

print("REGRESSION: Predict a continuous number")
print("  Examples: house price, temperature, stock price")
print("  Output: $325,000 or 72.5°F")

print("\nCLASSIFICATION: Predict a category")
print("  Examples: spam/not spam, disease/healthy, cat/dog")
print("  Output: 'Spam' or 'Cat'")

print("\nToday we focus on REGRESSION (predicting numbers)")

# ============================================
# CONCEPT 5: THE LEARNING PROCESS
# ============================================

print("\n" + "=" * 60)
print("5. HOW MODELS LEARN")
print("-" * 60)

print("1. Start with random guesses")
print("2. Make predictions")
print("3. Calculate error (how wrong we are)")
print("4. Adjust to reduce error")
print("5. Repeat steps 2-4 many times")
print("6. Eventually, predictions become accurate!")

print("\nThis is called 'Training' or 'Fitting' the model")

# Simple visualization
print("\n" + "=" * 60)
print("VISUAL EXAMPLE: LINEAR REGRESSION")
print("-" * 60)

# Create simple linear relationship with noise
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2 * X + 5 + np.random.randn(50) * 2  # y = 2x + 5 + noise

# Plot
plt.figure(figsize=(12, 5))

# Before learning
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6, s=50)
plt.plot(X, X, 'r--', linewidth=2, label='Random guess (wrong)')
plt.title('Before Learning', fontsize=14, fontweight='bold')
plt.xlabel('X (Input)', fontsize=12)
plt.ylabel('y (Output)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# After learning
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.6, s=50, label='Actual data')
plt.plot(X, 2*X + 5, 'g-', linewidth=3, label='Learned pattern')
plt.title('After Learning', fontsize=14, fontweight='bold')
plt.xlabel('X (Input)', fontsize=12)
plt.ylabel('y (Output)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/23_ml_learning_process.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: plots/23_ml_learning_process.png")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("1. ML learns patterns from data automatically")
print("2. Features (X) → Model → Predictions (y)")
print("3. Train on some data, test on different data")
print("4. Regression predicts numbers, Classification predicts categories")
print("5. Learning = Adjusting to minimize prediction errors")

print("\n✅ You now understand ML fundamentals!")
print("Next: Build your first model!")