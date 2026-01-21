"""
Day 8: Random Forests
Ensemble learning for superior performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RANDOM FORESTS: ENSEMBLE LEARNING")
print("=" * 60)

# ============================================
# CONCEPT: WHY RANDOM FORESTS?
# ============================================

print("\n1. THE ENSEMBLE CONCEPT")
print("-" * 60)

print("""
PROBLEM: Single decision tree is unstable
  • Small data change → Completely different tree
  • Tends to overfit
  
SOLUTION: Random Forest = Many Trees Voting Together!

HOW IT WORKS:
  1. Create 100 different decision trees
  2. Each tree sees random subset of data
  3. Each tree uses random subset of features
  4. All trees vote on final prediction
  5. Majority wins!

ANALOGY:
  One doctor's diagnosis → Risky
  100 doctors voting → More reliable!

This is called "Ensemble Learning"
""")

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("\n2. PREPARING TITANIC DATA")
print("-" * 60)

df = pd.read_csv('data/titanic.csv')

# Clean and engineer features
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Enhanced feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify titles
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
    'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
    'Capt': 'Rare', 'Ms': 'Miss'
}
df['Title'] = df['Title'].map(title_mapping)

# Age and Fare groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
df['FareGroup'] = pd.qcut(df['Fare'].rank(method='first'), q=4,
                          labels=['Low', 'Medium', 'High', 'VeryHigh'])

# Deck from Cabin
df['Deck'] = df['Cabin'].str[0]
df['Deck'].fillna('Unknown', inplace=True)

print(f"✅ Dataset prepared: {len(df)} passengers")

# Select features
feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 
                'FareGroup', 'Deck']

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['Survived']

print(f"Features created: {X.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================
# COMPARE: SINGLE TREE VS RANDOM FOREST
# ============================================

print("\n" + "=" * 60)
print("3. SINGLE TREE VS RANDOM FOREST")
print("=" * 60)

# Single Decision Tree
single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)

tree_train_acc = accuracy_score(y_train, single_tree.predict(X_train))
tree_test_acc = accuracy_score(y_test, single_tree.predict(X_test))

print("\nSINGLE DECISION TREE (depth=10):")
print(f"  Training Accuracy: {tree_train_acc:.4f}")
print(f"  Test Accuracy: {tree_test_acc:.4f}")
print(f"  Overfitting: {tree_train_acc - tree_test_acc:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

print("\nRANDOM FOREST (100 trees, depth=10):")
print(f"  Training Accuracy: {rf_train_acc:.4f}")
print(f"  Test Accuracy: {rf_test_acc:.4f}")
print(f"  Overfitting: {rf_train_acc - rf_test_acc:.4f}")

print(f"\n🏆 IMPROVEMENT:")
print(f"  Test Accuracy: +{(rf_test_acc - tree_test_acc)*100:.2f}%")
print(f"  Less Overfitting: {(tree_train_acc - tree_test_acc) - (rf_train_acc - rf_test_acc):.4f}")

# ============================================
# TUNING NUMBER OF TREES
# ============================================

print("\n" + "=" * 60)
print("4. FINDING OPTIMAL NUMBER OF TREES")
print("=" * 60)

n_estimators_range = [1, 10, 25, 50, 100, 200, 500]
results = []

for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
    rf_temp.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, rf_temp.predict(X_train))
    test_acc = accuracy_score(y_test, rf_temp.predict(X_test))
    
    results.append({
        'n_trees': n,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    print(f"  {n:3d} trees → Train: {train_acc:.4f}, Test: {test_acc:.4f}")

results_df = pd.DataFrame(results)

# ============================================
# FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 60)
print("5. FEATURE IMPORTANCE FROM RANDOM FOREST")
print("=" * 60)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ============================================
# MODEL EVALUATION
# ============================================

print("\n" + "=" * 60)
print("6. DETAILED EVALUATION")
print("=" * 60)

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation:")
print(f"  Scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("7. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest Analysis', fontsize=18, fontweight='bold')

# Plot 1: Tree vs Forest Comparison
ax1 = axes[0, 0]
models = ['Single Tree', 'Random Forest']
train_scores = [tree_train_acc, rf_train_acc]
test_scores = [tree_test_acc, rf_test_acc]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, train_scores, width, label='Train',
               alpha=0.8, color='skyblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test',
               alpha=0.8, color='lightcoral', edgecolor='black')

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Single Tree vs Random Forest', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.set_ylim(0.7, 1.0)
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Number of Trees Impact
ax2 = axes[0, 1]
ax2.plot(results_df['n_trees'], results_df['train_acc'], 'o-',
        linewidth=3, markersize=8, label='Training', color='blue')
ax2.plot(results_df['n_trees'], results_df['test_acc'], 's-',
        linewidth=3, markersize=8, label='Test', color='red')

ax2.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Impact of Number of Trees', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xscale('log')

# Plot 3: Feature Importance
ax3 = axes[1, 0]
top_15 = feature_importance.head(15)
bars = ax3.barh(range(len(top_15)), top_15['Importance'],
               color='lightgreen', edgecolor='black')
ax3.set_yticks(range(len(top_15)))
ax3.set_yticklabels(top_15['Feature'])
ax3.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax3.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, top_15['Importance'])):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

# Plot 4: Confusion Matrix
ax4 = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax4,
           xticklabels=['Died', 'Survived'],
           yticklabels=['Died', 'Survived'],
           annot_kws={'fontsize': 14})
ax4.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/35_random_forest_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/35_random_forest_analysis.png")

# ============================================
# OUT-OF-BAG (OOB) SCORE
# ============================================

print("\n" + "=" * 60)
print("8. OUT-OF-BAG VALIDATION")
print("=" * 60)

rf_oob = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.4f}")
print("""
OUT-OF-BAG SCORING:
  • Each tree trained on ~63% of data (bootstrap sample)
  • Remaining ~37% used for validation
  • Free cross-validation without separate test set!
  • OOB score ≈ cross-validation score
""")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("RANDOM FOREST KEY TAKEAWAYS")
print("=" * 60)

print(f"""
WHY RANDOM FORESTS WIN:
  ✓ Combines many trees → More stable
  ✓ Reduces overfitting compared to single tree
  ✓ Better generalization
  ✓ Built-in feature importance
  ✓ Handles missing data well
  ✓ Works on large datasets
  ✓ Minimal hyperparameter tuning needed

OUR RESULTS:
  • Test Accuracy: {rf_test_acc:.2%}
  • Better than single tree by {(rf_test_acc - tree_test_acc)*100:.1f}%
  • 100 trees is sweet spot (more doesn't help much)
  • Most important: Sex, Title, Fare, Age

WHEN TO USE:
  ✓ Tabular data (structured data)
  ✓ When interpretability is less critical
  ✓ When you need high accuracy
  ✓ Classification or regression
  
WHEN NOT TO USE:
  ✗ Very high-dimensional data (too slow)
  ✗ When you need to explain each decision
  ✗ Real-time predictions (slower than simple models)
""")

print("\n✅ Random Forests mastery complete!")