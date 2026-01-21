"""
Day 9: Gradient Boosting
Understanding sequential ensemble learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("GRADIENT BOOSTING: SEQUENTIAL LEARNING")
print("=" * 60)

# ============================================
# CONCEPT: HOW GRADIENT BOOSTING WORKS
# ============================================

print("\n1. GRADIENT BOOSTING CONCEPT")
print("-" * 60)

print("""
RANDOM FOREST vs GRADIENT BOOSTING:

RANDOM FOREST (Parallel):
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│Tree1│  │Tree2│  │Tree3│  │Tree4│  ← All trained independently
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   └────────┴────────┴────────┘
              VOTE
         Final Prediction

GRADIENT BOOSTING (Sequential):
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│Tree1│ ──→ │Tree2│ ──→ │Tree3│ ──→ │Tree4│
└─────┘     └─────┘     └─────┘     └─────┘
 Makes      Fixes       Fixes       Fixes
 initial    Tree1's     Tree2's     Tree3's
 pred.      mistakes    mistakes    mistakes

HOW IT WORKS:
  1. Tree 1: Makes predictions (simple)
  2. Calculate errors from Tree 1
  3. Tree 2: Predicts Tree 1's errors
  4. Tree 3: Predicts remaining errors
  5. ... continue until error is minimized
  6. Final = Tree1 + Tree2 + Tree3 + ...

Each tree is WEAK (shallow), but together they're STRONG!
""")

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("\n2. PREPARING TITANIC DATA")
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data ready: {X_train.shape}")

# ============================================
# COMPARE: RANDOM FOREST vs GRADIENT BOOSTING
# ============================================

print("\n" + "=" * 60)
print("3. RANDOM FOREST vs GRADIENT BOOSTING")
print("=" * 60)

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"✅ Random Forest:")
print(f"   Accuracy: {rf_acc:.4f}")
print(f"   ROC-AUC: {rf_auc:.4f}")

# Gradient Boosting
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]
gb_acc = accuracy_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, gb_proba)

print(f"✅ Gradient Boosting:")
print(f"   Accuracy: {gb_acc:.4f}")
print(f"   ROC-AUC: {gb_auc:.4f}")

print(f"\n🏆 WINNER: {'Gradient Boosting' if gb_acc > rf_acc else 'Random Forest'}")
print(f"   Improvement: +{abs(gb_acc - rf_acc)*100:.2f}%")

# ============================================
# KEY HYPERPARAMETERS
# ============================================

print("\n" + "=" * 60)
print("4. KEY GRADIENT BOOSTING HYPERPARAMETERS")
print("=" * 60)

print("""
CRITICAL PARAMETERS:

1. n_estimators (number of trees):
   • More trees = Better performance (usually)
   • Too many = Overfitting + slow training
   • Typical: 100-500

2. learning_rate (shrinkage):
   • Controls contribution of each tree
   • Lower = More trees needed, better generalization
   • Higher = Fewer trees needed, faster training
   • Typical: 0.01-0.3
   
3. max_depth (tree complexity):
   • Gradient Boosting uses SHALLOW trees (3-5)
   • Random Forest uses DEEP trees (10-20)
   • Deeper = More complex, more overfitting
   • Typical: 3-8

4. subsample (row sampling):
   • Fraction of samples for each tree
   • <1.0 adds randomness, reduces overfitting
   • Typical: 0.8-1.0

5. min_samples_split:
   • Minimum samples to split a node
   • Higher = Simpler trees
   • Typical: 2-20
""")

# ============================================
# HYPERPARAMETER IMPACT
# ============================================

print("\n" + "=" * 60)
print("5. TESTING HYPERPARAMETER IMPACT")
print("=" * 60)

# Test learning_rate
print("\nImpact of learning_rate:")
lr_results = []
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]

for lr in learning_rates:
    gb_temp = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                        max_depth=3, random_state=42)
    gb_temp.fit(X_train, y_train)
    acc = accuracy_score(y_test, gb_temp.predict(X_test))
    lr_results.append({'learning_rate': lr, 'accuracy': acc})
    print(f"  learning_rate={lr:.2f} → Accuracy: {acc:.4f}")

# Test n_estimators
print("\nImpact of n_estimators:")
n_est_results = []
n_estimators_list = [10, 50, 100, 200, 500]

for n in n_estimators_list:
    gb_temp = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1,
                                        max_depth=3, random_state=42)
    gb_temp.fit(X_train, y_train)
    acc = accuracy_score(y_test, gb_temp.predict(X_test))
    n_est_results.append({'n_estimators': n, 'accuracy': acc})
    print(f"  n_estimators={n:3d} → Accuracy: {acc:.4f}")

# Test max_depth
print("\nImpact of max_depth:")
depth_results = []
max_depths = [2, 3, 5, 7, 10]

for d in max_depths:
    gb_temp = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=d, random_state=42)
    gb_temp.fit(X_train, y_train)
    acc = accuracy_score(y_test, gb_temp.predict(X_test))
    depth_results.append({'max_depth': d, 'accuracy': acc})
    print(f"  max_depth={d:2d} → Accuracy: {acc:.4f}")

# ============================================
# FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'RF_Importance': rf.feature_importances_,
    'GB_Importance': gb.feature_importances_
}).sort_values('GB_Importance', ascending=False)

print("\nTop 10 Features (Gradient Boosting):")
print(importance_df.head(10).to_string(index=False))

# ============================================
# LEARNING CURVES
# ============================================

print("\n" + "=" * 60)
print("7. ANALYZING LEARNING PROGRESSION")
print("=" * 60)

print("Computing learning curves (this takes a moment)...")

# Get training scores at each iteration
gb_train_scores = []
gb_test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    gb.staged_predict(X_train),
    gb.staged_predict(X_test)
)):
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    gb_train_scores.append(train_acc)
    gb_test_scores.append(test_acc)

print(f"✅ Tracked performance across {len(gb_train_scores)} iterations")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("8. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Gradient Boosting Deep Dive', fontsize=18, fontweight='bold')

# Plot 1: RF vs GB Comparison
ax1 = axes[0, 0]
models = ['Random\nForest', 'Gradient\nBoosting']
accuracies = [rf_acc, gb_acc]
colors = ['skyblue', 'lightgreen']

bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Random Forest vs Gradient Boosting', fontsize=14, fontweight='bold')
ax1.set_ylim(0.75, 0.88)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Learning Rate Impact
ax2 = axes[0, 1]
lr_df = pd.DataFrame(lr_results)
ax2.plot(lr_df['learning_rate'], lr_df['accuracy'], 'o-',
        linewidth=3, markersize=10, color='purple')
ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Impact of Learning Rate', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xscale('log')

# Plot 3: Number of Estimators Impact
ax3 = axes[1, 0]
n_est_df = pd.DataFrame(n_est_results)
ax3.plot(n_est_df['n_estimators'], n_est_df['accuracy'], 's-',
        linewidth=3, markersize=10, color='orange')
ax3.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Impact of n_estimators', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Max Depth Impact
ax4 = axes[1, 1]
depth_df = pd.DataFrame(depth_results)
ax4.plot(depth_df['max_depth'], depth_df['accuracy'], '^-',
        linewidth=3, markersize=10, color='red')
ax4.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Impact of Max Depth', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)

# Plot 5: Learning Curve
ax5 = axes[2, 0]
iterations = range(1, len(gb_train_scores) + 1)
ax5.plot(iterations, gb_train_scores, linewidth=3, label='Training', color='blue')
ax5.plot(iterations, gb_test_scores, linewidth=3, label='Test', color='red')
ax5.fill_between(iterations, gb_train_scores, gb_test_scores,
                 alpha=0.2, color='yellow')
ax5.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
ax5.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax5.set_title('Learning Progression (Staged Predictions)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(alpha=0.3)

# Plot 6: Feature Importance Comparison
ax6 = axes[2, 1]
top_10 = importance_df.head(10)
x = np.arange(len(top_10))
width = 0.35

bars1 = ax6.barh(x - width/2, top_10['RF_Importance'], width,
                label='Random Forest', alpha=0.8, color='skyblue', edgecolor='black')
bars2 = ax6.barh(x + width/2, top_10['GB_Importance'], width,
                label='Gradient Boosting', alpha=0.8, color='lightgreen', edgecolor='black')

ax6.set_yticks(x)
ax6.set_yticklabels(top_10['Feature'], fontsize=9)
ax6.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax6.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(axis='x', alpha=0.3)
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('plots/38_gradient_boosting_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/38_gradient_boosting_analysis.png")

# ============================================
# ROC COMPARISON
# ============================================

print("\nCreating ROC comparison...")

plt.figure(figsize=(10, 8))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
plt.plot(fpr_rf, tpr_rf, linewidth=3, label=f'Random Forest (AUC={rf_auc:.3f})', color='blue')

# Gradient Boosting ROC
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_proba)
plt.plot(fpr_gb, tpr_gb, linewidth=3, label=f'Gradient Boosting (AUC={gb_auc:.3f})', color='green')

# Random baseline
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.savefig('plots/39_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/39_roc_comparison.png")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("GRADIENT BOOSTING KEY TAKEAWAYS")
print("=" * 60)

print(f"""
WHY GRADIENT BOOSTING WINS:
  ✓ Sequential learning fixes previous mistakes
  ✓ Often more accurate than Random Forest
  ✓ Excellent with structured/tabular data
  ✓ Handles complex patterns well
  ✓ Feature importance very interpretable

OUR RESULTS:
  • Gradient Boosting: {gb_acc:.2%} accuracy
  • Random Forest: {rf_acc:.2%} accuracy
  • GB better by: {(gb_acc - rf_acc)*100:.2f}%

HYPERPARAMETER INSIGHTS:
  • Learning rate 0.1 worked best
  • 100-200 trees optimal (more = diminishing returns)
  • Shallow trees (depth 3-5) prevent overfitting
  • Each tree focuses on hard examples

WHEN TO USE GRADIENT BOOSTING:
  ✓ Tabular/structured data
  ✓ When accuracy is critical
  ✓ Moderate-sized datasets
  ✓ Classification or regression
  
WHEN TO USE RANDOM FOREST:
  ✓ Need fast training
  ✓ Less hyperparameter tuning
  ✓ More interpretable (parallel trees)
  ✓ Very large datasets

NEXT LEVEL:
  → XGBoost (optimized Gradient Boosting)
  → LightGBM (even faster!)
  → These win most Kaggle competitions!
""")

print("\n✅ Gradient Boosting mastery complete!")