"""
Day 9: XGBoost & LightGBM
The algorithms that win Kaggle competitions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("XGBOOST & LIGHTGBM: COMPETITION WINNERS")
print("=" * 60)

# ============================================
# WHAT ARE XGBOOST AND LIGHTGBM?
# ============================================

print("\n1. UNDERSTANDING THE CHAMPIONS")
print("-" * 60)

print("""
GRADIENT BOOSTING EVOLUTION:

1. scikit-learn GradientBoosting (2007):
   ✓ Good baseline
   ✗ Slow on large datasets
   ✗ Limited features

2. XGBoost (2014):
   ✓ 10x faster than sklearn
   ✓ Regularization to prevent overfitting
   ✓ Handles missing values automatically
   ✓ Built-in cross-validation
   ✓ Parallel processing
   🏆 Dominated Kaggle 2015-2017

3. LightGBM (2017):
   ✓ Even faster than XGBoost
   ✓ Lower memory usage
   ✓ Better with large datasets (millions of rows)
   ✓ Leaf-wise tree growth (vs level-wise)
   🏆 Current Kaggle favorite

WHY THEY WIN:
  • Extreme optimization
  • Advanced regularization
  • GPU support
  • Production-ready
""")

# ============================================
# LOAD AND PREPARE DATA
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

title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
df['Title'] = df['Title'].map(title_map).fillna('Rare')

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=[0, 1, 2, 3, 4])
df['FareGroup'] = pd.qcut(df['Fare'].rank(method='first'), q=4, labels=[0, 1, 2, 3])

# Features
X = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                        'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']],
                   drop_first=True)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data ready: {X_train.shape}")

# ============================================
# TRAIN ALL BOOSTING ALGORITHMS
# ============================================

print("\n" + "=" * 60)
print("3. COMPARING ALL GRADIENT BOOSTING METHODS")
print("=" * 60)

results = []

# 1. sklearn Gradient Boosting
print("\n1. Training sklearn GradientBoosting...")
start_time = time.time()

gb_sklearn = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_sklearn.fit(X_train, y_train)

gb_sklearn_time = time.time() - start_time
gb_sklearn_pred = gb_sklearn.predict(X_test)
gb_sklearn_proba = gb_sklearn.predict_proba(X_test)[:, 1]
gb_sklearn_acc = accuracy_score(y_test, gb_sklearn_pred)
gb_sklearn_auc = roc_auc_score(y_test, gb_sklearn_proba)

print(f"✅ sklearn GB: {gb_sklearn_time:.2f}s")
print(f"   Accuracy: {gb_sklearn_acc:.4f} | AUC: {gb_sklearn_auc:.4f}")

results.append({
    'Model': 'sklearn GB',
    'Accuracy': gb_sklearn_acc,
    'ROC-AUC': gb_sklearn_auc,
    'Time (s)': gb_sklearn_time
})

# 2. XGBoost
print("\n2. Training XGBoost...")
start_time = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_time = time.time() - start_time
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"✅ XGBoost: {xgb_time:.2f}s")
print(f"   Accuracy: {xgb_acc:.4f} | AUC: {xgb_auc:.4f}")
print(f"   Speedup: {gb_sklearn_time/xgb_time:.1f}x faster!")

results.append({
    'Model': 'XGBoost',
    'Accuracy': xgb_acc,
    'ROC-AUC': xgb_auc,
    'Time (s)': xgb_time
})

# 3. LightGBM
print("\n3. Training LightGBM...")
start_time = time.time()

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

lgb_time = time.time() - start_time
lgb_pred = lgb_model.predict(X_test)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_auc = roc_auc_score(y_test, lgb_proba)

print(f"✅ LightGBM: {lgb_time:.2f}s")
print(f"   Accuracy: {lgb_acc:.4f} | AUC: {lgb_auc:.4f}")
print(f"   Speedup: {gb_sklearn_time/lgb_time:.1f}x faster!")

results.append({
    'Model': 'LightGBM',
    'Accuracy': lgb_acc,
    'ROC-AUC': lgb_auc,
    'Time (s)': lgb_time
})

# 4. Random Forest (for comparison)
print("\n4. Training Random Forest...")
start_time = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

rf_time = time.time() - start_time
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"✅ Random Forest: {rf_time:.2f}s")
print(f"   Accuracy: {rf_acc:.4f} | AUC: {rf_auc:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_acc,
    'ROC-AUC': rf_auc,
    'Time (s)': rf_time
})

# ============================================
# RESULTS COMPARISON
# ============================================

print("\n" + "=" * 60)
print("COMPREHENSIVE COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\n" + results_df.to_string(index=False))

best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.4f}")
print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")
print(f"   Time: {best_model['Time (s)']:.2f}s")

# ============================================
# XGBOOST ADVANCED FEATURES
# ============================================

print("\n" + "=" * 60)
print("4. XGBOOST ADVANCED FEATURES")
print("=" * 60)

print("\nTuned XGBoost with regularization...")

xgb_tuned = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    gamma=0.1,  # Regularization
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42,
    eval_metric='logloss'
)

xgb_tuned.fit(X_train, y_train)

xgb_tuned_pred = xgb_tuned.predict(X_test)
xgb_tuned_proba = xgb_tuned.predict_proba(X_test)[:, 1]
xgb_tuned_acc = accuracy_score(y_test, xgb_tuned_pred)
xgb_tuned_auc = roc_auc_score(y_test, xgb_tuned_proba)

print(f"✅ Tuned XGBoost:")
print(f"   Accuracy: {xgb_tuned_acc:.4f}")
print(f"   ROC-AUC: {xgb_tuned_auc:.4f}")
print(f"   Improvement: +{(xgb_tuned_acc - xgb_acc)*100:.2f}%")

# ============================================
# FEATURE IMPORTANCE COMPARISON
# ============================================

print("\n" + "=" * 60)
print("5. FEATURE IMPORTANCE COMPARISON")
print("=" * 60)

importance_comparison = pd.DataFrame({
    'Feature': X.columns,
    'XGBoost': xgb_model.feature_importances_,
    'LightGBM': lgb_model.feature_importances_,
    'RandomForest': rf.feature_importances_
}).sort_values('XGBoost', ascending=False)

print("\nTop 10 Features:")
print(importance_comparison.head(10).to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("6. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost & LightGBM Analysis', fontsize=18, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
models = results_df['Model'].values
accuracies = results_df['Accuracy'].values
colors_list = ['lightblue', 'lightgreen', 'lightcoral', 'gold']

bars = ax1.barh(models, accuracies, color=colors_list[:len(models)], 
               edgecolor='black', linewidth=2)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0.78, 0.87)
ax1.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, accuracies):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=11, fontweight='bold')

# Plot 2: Training Time Comparison
ax2 = axes[0, 1]
times = results_df['Time (s)'].values

bars = ax2.bar(models, times, color=colors_list[:len(models)],
              edgecolor='black', linewidth=2)
ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Feature Importance (Top 12)
ax3 = axes[1, 0]
top_12 = importance_comparison.head(12)
x = np.arange(len(top_12))
width = 0.25

ax3.barh(x - width, top_12['XGBoost'], width, label='XGBoost',
        alpha=0.8, color='skyblue', edgecolor='black')
ax3.barh(x, top_12['LightGBM'], width, label='LightGBM',
        alpha=0.8, color='lightgreen', edgecolor='black')
ax3.barh(x + width, top_12['RandomForest'], width, label='Random Forest',
        alpha=0.8, color='lightcoral', edgecolor='black')

ax3.set_yticks(x)
ax3.set_yticklabels(top_12['Feature'], fontsize=9)
ax3.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax3.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# Plot 4: Accuracy vs Speed
ax4 = axes[1, 1]

for i, row in results_df.iterrows():
    ax4.scatter(row['Time (s)'], row['Accuracy'], s=300, alpha=0.7,
               label=row['Model'], edgecolors='black', linewidth=2)

ax4.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# Annotate best model
best_row = results_df.iloc[0]
ax4.annotate('Best Model!', 
            xy=(best_row['Time (s)'], best_row['Accuracy']),
            xytext=(best_row['Time (s)'] + 0.1, best_row['Accuracy'] - 0.01),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

plt.tight_layout()
plt.savefig('plots/40_xgboost_lightgbm_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/40_xgboost_lightgbm_comparison.png")

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print(f"""
PERFORMANCE SUMMARY:
  🏆 Best Accuracy: {results_df.iloc[0]['Model']} ({results_df.iloc[0]['Accuracy']:.4f})
  ⚡ Fastest: {results_df.loc[results_df['Time (s)'].idxmin(), 'Model']} ({results_df['Time (s)'].min():.2f}s)
  
SPEED COMPARISON:
  • sklearn GB: {gb_sklearn_time:.2f}s (baseline)
  • XGBoost: {xgb_time:.2f}s ({gb_sklearn_time/xgb_time:.1f}x faster!)
  • LightGBM: {lgb_time:.2f}s ({gb_sklearn_time/lgb_time:.1f}x faster!)
  
WHEN TO USE EACH:

XGBoost:
  ✓ Most Kaggle competitions
  ✓ When accuracy is critical
  ✓ Medium-sized datasets
  ✓ Need regularization
  ✓ Production ML systems

LightGBM:
  ✓ Large datasets (millions of rows)
  ✓ Need fastest training
  ✓ Memory constraints
  ✓ Real-time applications
  
sklearn GradientBoosting:
  ✓ Educational purposes
  ✓ Simple baselines
  ✓ No external dependencies
  
Random Forest:
  ✓ Quick prototypes
  ✓ Less tuning needed
  ✓ More interpretable

ADVANCED FEATURES (XGBoost/LightGBM):
  • Built-in cross-validation
  • Early stopping
  • GPU acceleration
  • Custom loss functions
  • Missing value handling
  • Categorical feature support
  • Regularization (L1 & L2)
""")

print("\n✅ XGBoost & LightGBM mastery complete!")