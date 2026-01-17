"""
Day 8: Hyperparameter Tuning
Finding the best model configuration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.metrics import accuracy_score, make_scorer
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# ============================================
# WHAT ARE HYPERPARAMETERS?
# ============================================

print("\n1. UNDERSTANDING HYPERPARAMETERS")
print("-" * 60)

print("""
PARAMETERS: Learned from data during training
  Example: Decision tree split points, feature weights
  
HYPERPARAMETERS: Set BEFORE training
  Example: max_depth, n_estimators, learning_rate
  
KEY RANDOM FOREST HYPERPARAMETERS:
  • n_estimators: Number of trees (50, 100, 200, ...)
  • max_depth: Maximum tree depth (5, 10, 20, None)
  • min_samples_split: Min samples to split node (2, 5, 10)
  • min_samples_leaf: Min samples in leaf (1, 2, 4)
  • max_features: Features to consider per split ('sqrt', 'log2', None)
  
GOAL: Find best combination for YOUR data
""")

# ============================================
# PREPARE DATA
# ============================================

print("\n2. PREPARING DATA")
print("-" * 60)

df = pd.read_csv('data/titanic.csv')

# Feature engineering
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify titles
title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
df['Title'] = df['Title'].map(title_map).fillna('Rare')

# Features
X = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                        'FamilySize', 'IsAlone', 'Title']], drop_first=True)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data ready: {X_train.shape}")

# ============================================
# BASELINE MODEL
# ============================================

print("\n" + "=" * 60)
print("3. BASELINE MODEL (Default Parameters)")
print("=" * 60)

baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)

baseline_score = accuracy_score(y_test, baseline.predict(X_test))
print(f"Baseline Accuracy: {baseline_score:.4f}")
print(f"Default parameters: {baseline.get_params()}")

# ============================================
# METHOD 1: MANUAL TUNING
# ============================================

print("\n" + "=" * 60)
print("4. METHOD 1: MANUAL TUNING")
print("=" * 60)

print("Testing different n_estimators values...")

manual_results = []
n_est_values = [10, 50, 100, 200, 500]

for n in n_est_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    score = accuracy_score(y_test, rf.predict(X_test))
    manual_results.append({'n_estimators': n, 'accuracy': score})
    print(f"  n_estimators={n:3d} → Accuracy: {score:.4f}")

manual_df = pd.DataFrame(manual_results)
best_manual = manual_df.loc[manual_df['accuracy'].idxmax()]
print(f"\n✅ Best: n_estimators={int(best_manual['n_estimators'])} → {best_manual['accuracy']:.4f}")

# ============================================
# METHOD 2: GRID SEARCH
# ============================================

print("\n" + "=" * 60)
print("5. METHOD 2: GRID SEARCH (Exhaustive)")
print("=" * 60)

print("Searching through parameter grid...")
print("This tests EVERY combination!\n")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combinations = (len(param_grid['n_estimators']) *
                     len(param_grid['max_depth']) *
                     len(param_grid['min_samples_split']) *
                     len(param_grid['min_samples_leaf']))

print(f"Total combinations to test: {total_combinations}")
print(f"With 5-fold CV: {total_combinations * 5} model trainings!")

start_time = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time

print(f"\n✅ Grid Search complete in {grid_time:.2f} seconds")
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Test on holdout set
grid_test_score = accuracy_score(y_test, grid_search.predict(X_test))
print(f"Test set score: {grid_test_score:.4f}")

# ============================================
# METHOD 3: RANDOMIZED SEARCH
# ============================================

print("\n" + "=" * 60)
print("6. METHOD 3: RANDOMIZED SEARCH (Faster)")
print("=" * 60)

print("Randomly sampling parameter combinations...")
print("Tests fewer combinations, still finds good results!\n")

# Define parameter distributions
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', None]
}

print(f"Trying 50 random combinations (out of thousands possible)")

start_time = time.time()

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of random combinations
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

random_time = time.time() - start_time

print(f"\n✅ Random Search complete in {random_time:.2f} seconds")
print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Test on holdout set
random_test_score = accuracy_score(y_test, random_search.predict(X_test))
print(f"Test set score: {random_test_score:.4f}")

# ============================================
# COMPARISON
# ============================================

print("\n" + "=" * 60)
print("7. METHOD COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Method': ['Baseline', 'Manual', 'Grid Search', 'Random Search'],
    'Test Accuracy': [baseline_score, best_manual['accuracy'],
                     grid_test_score, random_test_score],
    'Time (seconds)': [0, 5, grid_time, random_time]
})

print(comparison.to_string(index=False))

best_method = comparison.loc[comparison['Test Accuracy'].idxmax(), 'Method']
best_accuracy = comparison['Test Accuracy'].max()

print(f"\n🏆 WINNER: {best_method} with {best_accuracy:.4f} accuracy")

# ============================================
# ANALYZE GRID SEARCH RESULTS
# ============================================

print("\n" + "=" * 60)
print("8. ANALYZING PARAMETER IMPACT")
print("=" * 60)

# Convert results to DataFrame
grid_results = pd.DataFrame(grid_search.cv_results_)

# Analyze n_estimators impact
print("\nImpact of n_estimators:")
for n in param_grid['n_estimators']:
    mask = grid_results['param_n_estimators'] == n
    mean_score = grid_results.loc[mask, 'mean_test_score'].mean()
    print(f"  n_estimators={n:3d} → Mean score: {mean_score:.4f}")

# Analyze max_depth impact
print("\nImpact of max_depth:")
for d in param_grid['max_depth']:
    mask = grid_results['param_max_depth'] == d
    mean_score = grid_results.loc[mask, 'mean_test_score'].mean()
    depth_str = str(d) if d else "None"
    print(f"  max_depth={depth_str:>4s} → Mean score: {mean_score:.4f}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("9. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hyperparameter Tuning Analysis', fontsize=18, fontweight='bold')

# Plot 1: Method Comparison
ax1 = axes[0, 0]
bars = ax1.bar(comparison['Method'], comparison['Test Accuracy'],
              color=['gray', 'skyblue', 'lightgreen', 'lightcoral'],
              edgecolor='black', linewidth=2)
ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Tuning Method Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0.75, 0.88)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, comparison['Test Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Time Comparison
ax2 = axes[0, 1]
time_data = comparison[comparison['Time (seconds)'] > 0]
bars = ax2.barh(time_data['Method'], time_data['Time (seconds)'],
               color=['skyblue', 'lightgreen', 'lightcoral'],
               edgecolor='black')
ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Computation Time', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, time_data['Time (seconds)']):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}s', va='center', fontsize=10)

# Plot 3: n_estimators Impact (Grid Search)
ax3 = axes[1, 0]
n_est_scores = []
for n in param_grid['n_estimators']:
    mask = grid_results['param_n_estimators'] == n
    scores = grid_results.loc[mask, 'mean_test_score']
    n_est_scores.append(scores.values)

bp = ax3.boxplot(n_est_scores, labels=param_grid['n_estimators'],
                patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
ax3.set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Impact of n_estimators', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: max_depth Impact
ax4 = axes[1, 1]
depth_labels = [str(d) if d else 'None' for d in param_grid['max_depth']]
depth_scores = []
for d in param_grid['max_depth']:
    mask = grid_results['param_max_depth'] == d
    scores = grid_results.loc[mask, 'mean_test_score']
    depth_scores.append(scores.values)

bp = ax4.boxplot(depth_scores, labels=depth_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax4.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax4.set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Impact of max_depth', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/36_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/36_hyperparameter_tuning.png")

# ============================================
# BEST PRACTICES
# ============================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING BEST PRACTICES")
print("=" * 60)

print(f"""
WHEN TO USE EACH METHOD:

1. MANUAL TUNING:
   ✓ Small datasets
   ✓ Understanding parameter impact
   ✓ Quick experiments
   
2. GRID SEARCH:
   ✓ Small parameter space
   ✓ When thoroughness > speed
   ✓ Final model optimization
   
3. RANDOMIZED SEARCH:
   ✓ Large parameter space
   ✓ When speed > thoroughness
   ✓ Initial exploration
   
OUR RESULTS:
  • Grid Search: {grid_time:.1f}s → {grid_test_score:.4f} accuracy
  • Random Search: {random_time:.1f}s → {random_test_score:.4f} accuracy
  • Random Search is {grid_time/random_time:.1f}x faster!
  
RECOMMENDATIONS:
  1. Start with Randomized Search (fast exploration)
  2. Narrow down promising regions
  3. Use Grid Search for fine-tuning
  4. Always use cross-validation
  5. Keep a holdout test set
""")

print("\n✅ Hyperparameter tuning mastery complete!")