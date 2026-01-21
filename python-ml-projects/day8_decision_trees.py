"""
Day 8: Decision Trees
Understanding tree-based classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DECISION TREES: HOW THEY WORK")
print("=" * 60)

# ============================================
# CONCEPT: HOW DECISION TREES WORK
# ============================================

print("\n1. DECISION TREE CONCEPT")
print("-" * 60)

print("""
A Decision Tree makes decisions like a flowchart:

Example: Titanic Survival
┌─────────────────┐
│  Sex = Male?    │
└────┬────────┬───┘
   YES       NO
    │         │
    v         v
  DIED    ┌────────────┐
          │ Class = 3? │
          └──┬─────┬───┘
           YES    NO
            │      │
            v      v
          DIED  SURVIVED

Each split asks a YES/NO question about features.
Tree keeps splitting until it reaches a decision (leaf).
""")

# ============================================
# SIMPLE EXAMPLE
# ============================================

print("\n2. SIMPLE DECISION TREE EXAMPLE")
print("-" * 60)

# Create simple dataset
np.random.seed(42)
data = {
    'Age': [5, 8, 15, 25, 35, 45, 55, 65, 12, 22, 38, 48],
    'Income': [0, 0, 20, 40, 60, 80, 90, 50, 10, 35, 75, 85],
    'Buy_Product': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]  # 0=No, 1=Yes
}
df_simple = pd.DataFrame(data)

print("Simple dataset: Will customer buy product?")
print(df_simple)

# Split data
X_simple = df_simple[['Age', 'Income']]
y_simple = df_simple['Buy_Product']

# Train simple tree
simple_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
simple_tree.fit(X_simple, y_simple)

print("\n✅ Simple tree trained (max_depth=2)")

# Visualize the tree
plt.figure(figsize=(16, 8))
plot_tree(simple_tree, 
          feature_names=['Age', 'Income'],
          class_names=['No Buy', 'Buy'],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title('Simple Decision Tree: Customer Purchase Prediction', 
          fontsize=16, fontweight='bold', pad=20)
plt.savefig('plots/32_simple_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/32_simple_decision_tree.png")

print("\nTree structure:")
print(f"  Root split: Income <= {simple_tree.tree_.threshold[0]:.1f}")
print("  This single split separates buyers from non-buyers!")

# ============================================
# TITANIC WITH DECISION TREE
# ============================================

print("\n" + "=" * 60)
print("3. DECISION TREE ON TITANIC DATASET")
print("=" * 60)

# Load Titanic data
df = pd.read_csv('data/titanic.csv')

# Prepare data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Select features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

print(f"Dataset: {len(df)} passengers")
print(f"Features: {X.columns.tolist()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining: {len(X_train)} | Test: {len(X_test)}")

# ============================================
# TRAIN DECISION TREES WITH DIFFERENT DEPTHS
# ============================================

print("\n" + "=" * 60)
print("4. COMPARING TREE DEPTHS")
print("=" * 60)

depths = [2, 3, 5, 10, None]  # None = unlimited depth
results = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    depth_str = str(depth) if depth else "Unlimited"
    
    results.append({
        'Max Depth': depth_str,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Num Leaves': tree.get_n_leaves(),
        'Tree Depth': tree.get_depth()
    })
    
    print(f"\nDepth {depth_str}:")
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f}")
    print(f"  Leaves: {tree.get_n_leaves()} | Actual Depth: {tree.get_depth()}")

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================
# OPTIMAL TREE (depth=5)
# ============================================

print("\n" + "=" * 60)
print("5. TRAINING OPTIMAL DECISION TREE")
print("=" * 60)

# Best performing depth
best_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
best_tree.fit(X_train, y_train)

y_pred = best_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Optimal tree trained (max_depth=5)")
print(f"Test Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_tree.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("6. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Decision Tree Analysis', fontsize=18, fontweight='bold')

# Plot 1: Depth Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
width = 0.35

ax1.bar(x_pos - width/2, results_df['Train Accuracy'], 
        width, label='Train', alpha=0.8, color='skyblue', edgecolor='black')
ax1.bar(x_pos + width/2, results_df['Test Accuracy'], 
        width, label='Test', alpha=0.8, color='lightcoral', edgecolor='black')

ax1.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy vs Tree Depth', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Max Depth'])
ax1.legend()
ax1.set_ylim(0.6, 1.0)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=accuracy, color='green', linestyle='--', 
            linewidth=2, label=f'Optimal (depth=5): {accuracy:.3f}')

# Plot 2: Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(10)
bars = ax2.barh(top_features['Feature'], top_features['Importance'],
                color='lightgreen', edgecolor='black')
ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, top_features['Importance']):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

# Plot 3: Confusion Matrix
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
           xticklabels=['Died', 'Survived'],
           yticklabels=['Died', 'Survived'],
           annot_kws={'fontsize': 14})
ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax3.set_title('Confusion Matrix (Depth=5)', fontsize=14, fontweight='bold')

# Plot 4: Overfitting Analysis
ax4 = axes[1, 1]
train_accs = results_df['Train Accuracy'].values
test_accs = results_df['Test Accuracy'].values
x_labels = results_df['Max Depth'].values

ax4.plot(range(len(x_labels)), train_accs, 'o-', linewidth=3, 
         markersize=10, label='Training', color='blue')
ax4.plot(range(len(x_labels)), test_accs, 's-', linewidth=3,
         markersize=10, label='Test', color='red')
ax4.fill_between(range(len(x_labels)), train_accs, test_accs, 
                 alpha=0.2, color='yellow', label='Overfitting Gap')

ax4.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(x_labels)))
ax4.set_xticklabels(x_labels)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/33_decision_tree_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/33_decision_tree_analysis.png")

# ============================================
# VISUALIZE OPTIMAL TREE
# ============================================

print("\nVisualizing optimal tree structure...")

plt.figure(figsize=(20, 10))
plot_tree(best_tree,
          feature_names=X.columns,
          class_names=['Died', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)  # Show first 3 levels for clarity
plt.title('Titanic Decision Tree (Depth=5, showing first 3 levels)',
          fontsize=18, fontweight='bold', pad=20)
plt.savefig('plots/34_titanic_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/34_titanic_decision_tree.png")

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "=" * 60)
print("KEY DECISION TREE INSIGHTS")
print("=" * 60)

print(f"""
ADVANTAGES:
  ✓ Easy to visualize and interpret
  ✓ No feature scaling needed
  ✓ Handles non-linear relationships
  ✓ Shows feature importance clearly
  ✓ Works with mixed data types

DISADVANTAGES:
  ✗ Prone to overfitting (unlimited depth)
  ✗ Unstable (small data changes = different tree)
  ✗ Biased toward features with many levels
  
OVERFITTING OBSERVED:
  • Depth=2: Underfit (too simple)
  • Depth=5: Just right (best test score)
  • Depth=Unlimited: Overfit (perfect train, worse test)

SOLUTION TO OVERFITTING:
  → Random Forests (next part!)
""")

print("\n✅ Decision Trees mastery complete!")