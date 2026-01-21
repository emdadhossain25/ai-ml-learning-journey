"""
Day 10: Model Interpretability (Without SHAP dependency)
Understanding model decisions using built-in methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MODEL INTERPRETABILITY")
print("=" * 60)

# ============================================
# WHY INTERPRETABILITY MATTERS
# ============================================

print("\n1. WHY MODEL INTERPRETABILITY IS CRITICAL")
print("-" * 60)

print("""
WHY WE NEED INTERPRETABILITY:
  • Trust: Users need to understand predictions
  • Debugging: Find what model learned (right or wrong)
  • Fairness: Detect bias in predictions
  • Compliance: Regulatory requirements (GDPR, etc.)
  • Improvement: Identify what features matter

INTERPRETABILITY METHODS (Without SHAP):
  1. Feature Importance (built-in)
  2. Permutation Importance
  3. Partial Dependence Plots
  4. Individual prediction breakdown
""")

# ============================================
# PREPARE DATA
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
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
df['Title'] = df['Title'].map(title_map).fillna('Rare')

df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
df['Age_Class'] = df['Age'] * df['Pclass']

# Prepare features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
           'FamilySize', 'IsAlone', 'Title', 'Fare_Per_Person', 'Age_Class']

X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data ready: {X_train.shape}")

# ============================================
# TRAIN MODELS
# ============================================

print("\n" + "=" * 60)
print("3. TRAINING MODELS")
print("=" * 60)

# Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")

# XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, 
                              learning_rate=0.1, random_state=42,
                              eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_accuracy = xgb_model.score(X_test, y_test)
print(f"✅ XGBoost Accuracy: {xgb_accuracy:.4f}")

# ============================================
# FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 60)
print("4. FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Built-in importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'RF_Importance': rf_model.feature_importances_,
    'XGB_Importance': xgb_model.feature_importances_
}).sort_values('XGB_Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(rf_importance.head(10).to_string(index=False))

# ============================================
# PERMUTATION IMPORTANCE
# ============================================

print("\n" + "=" * 60)
print("5. PERMUTATION IMPORTANCE")
print("=" * 60)

print("\nCalculating permutation importance...")
print("(This shows importance by shuffling each feature)")

perm_importance = permutation_importance(
    xgb_model, X_test, y_test, n_repeats=10, random_state=42
)

perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Permutation_Importance': perm_importance.importances_mean
}).sort_values('Permutation_Importance', ascending=False)

print("\nTop 10 by Permutation Importance:")
print(perm_importance_df.head(10).to_string(index=False))

# ============================================
# INDIVIDUAL PREDICTIONS
# ============================================

print("\n" + "=" * 60)
print("6. UNDERSTANDING INDIVIDUAL PREDICTIONS")
print("=" * 60)

# Example passenger who survived
survived_idx = y_test[y_test == 1].index[0]
survived_passenger = X_test.loc[survived_idx]
survived_pred_proba = xgb_model.predict_proba(survived_passenger.values.reshape(1, -1))[0, 1]

print("\nExample 1: Passenger who SURVIVED")
print(f"Prediction probability: {survived_pred_proba:.2%}")
print("\nKey features:")
for feat, val in survived_passenger.items():
    if val != 0:
        importance = xgb_model.feature_importances_[list(X.columns).index(feat)]
        print(f"  {feat}: {val:.2f} (importance: {importance:.4f})")

# Example passenger who died
died_idx = y_test[y_test == 0].index[0]
died_passenger = X_test.loc[died_idx]
died_pred_proba = xgb_model.predict_proba(died_passenger.values.reshape(1, -1))[0, 1]

print("\nExample 2: Passenger who DIED")
print(f"Prediction probability: {died_pred_proba:.2%}")
print("\nKey features:")
for feat, val in died_passenger.items():
    if val != 0:
        importance = xgb_model.feature_importances_[list(X.columns).index(feat)]
        print(f"  {feat}: {val:.2f} (importance: {importance:.4f})")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("7. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Interpretability Analysis', fontsize=18, fontweight='bold')

# Plot 1: Feature Importance Comparison
ax1 = axes[0, 0]
top_15 = rf_importance.head(15)
x = np.arange(len(top_15))
width = 0.35

bars1 = ax1.barh(x - width/2, top_15['RF_Importance'], width,
                label='Random Forest', alpha=0.8, color='skyblue', edgecolor='black')
bars2 = ax1.barh(x + width/2, top_15['XGB_Importance'], width,
                label='XGBoost', alpha=0.8, color='lightgreen', edgecolor='black')

ax1.set_yticks(x)
ax1.set_yticklabels(top_15['Feature'], fontsize=9)
ax1.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax1.set_title('Feature Importance Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Plot 2: Permutation Importance
ax2 = axes[0, 1]
top_15_perm = perm_importance_df.head(15)

bars = ax2.barh(range(len(top_15_perm)), top_15_perm['Permutation_Importance'],
               color='lightcoral', edgecolor='black')
ax2.set_yticks(range(len(top_15_perm)))
ax2.set_yticklabels(top_15_perm['Feature'], fontsize=9)
ax2.set_xlabel('Permutation Importance', fontsize=11, fontweight='bold')
ax2.set_title('Permutation Importance (XGBoost)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Plot 3: Feature Importance Methods Comparison
ax3 = axes[1, 0]

# Combine all importance methods for top 10 features
top_10_features = rf_importance.head(10)['Feature'].tolist()
comparison_data = []

for feat in top_10_features:
    rf_imp = rf_importance[rf_importance['Feature'] == feat]['RF_Importance'].values[0]
    xgb_imp = rf_importance[rf_importance['Feature'] == feat]['XGB_Importance'].values[0]
    perm_imp = perm_importance_df[perm_importance_df['Feature'] == feat]['Permutation_Importance'].values[0]
    
    comparison_data.append({
        'Feature': feat,
        'RF': rf_imp,
        'XGB': xgb_imp,
        'Perm': perm_imp
    })

comp_df = pd.DataFrame(comparison_data)

x = np.arange(len(comp_df))
width = 0.25

ax3.bar(x - width, comp_df['RF'], width, label='RF', alpha=0.8, color='skyblue', edgecolor='black')
ax3.bar(x, comp_df['XGB'], width, label='XGB', alpha=0.8, color='lightgreen', edgecolor='black')
ax3.bar(x + width, comp_df['Perm'], width, label='Perm', alpha=0.8, color='lightcoral', edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(comp_df['Feature'], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Importance', fontsize=11, fontweight='bold')
ax3.set_title('Importance Methods Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Decision Path Explanation
ax4 = axes[1, 1]
ax4.axis('off')

explanation = f"""
╔══════════════════════════════════════════════╗
║     HOW TO INTERPRET FEATURE IMPORTANCE      ║
╠══════════════════════════════════════════════╣
║                                              ║
║  🔍 BUILT-IN IMPORTANCE:                     ║
║     • Shows how much model uses feature      ║
║     • Based on training process              ║
║     • Fast to compute                        ║
║                                              ║
║  🎲 PERMUTATION IMPORTANCE:                  ║
║     • Shuffle feature → measure impact       ║
║     • Shows real-world importance            ║
║     • More reliable but slower               ║
║                                              ║
║  📊 TOP INSIGHTS:                            ║
║     • Sex is most important                  ║
║     • Class matters significantly            ║
║     • Age and Fare also important            ║
║                                              ║
║  ✨ SURVIVED PASSENGER:                      ║
║     • Probability: {survived_pred_proba:.1%}                        ║
║     • Likely female, young, high class       ║
║                                              ║
║  ⚠️  DIED PASSENGER:                         ║
║     • Probability: {died_pred_proba:.1%}                        ║
║     • Likely male, older, low class          ║
║                                              ║
║  💡 KEY TAKEAWAY:                            ║
║     Models learn patterns from data          ║
║     Understanding them builds trust          ║
║                                              ║
╚══════════════════════════════════════════════╝
"""

ax4.text(0.05, 0.5, explanation, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig('plots/44_model_interpretability.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/44_model_interpretability.png")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: MODEL INTERPRETABILITY")
print("=" * 60)

print(f"""
INTERPRETABILITY METHODS LEARNED:

1. BUILT-IN FEATURE IMPORTANCE:
   ✓ Fast to compute
   ✓ Shows model's internal weights
   ✓ Top feature: {rf_importance.iloc[0]['Feature']}

2. PERMUTATION IMPORTANCE:
   ✓ More reliable
   ✓ Shows real-world impact
   ✓ Slower but better
   ✓ Top feature: {perm_importance_df.iloc[0]['Feature']}

3. INDIVIDUAL PREDICTIONS:
   ✓ Explained specific decisions
   ✓ Survived: {survived_pred_proba:.1%} probability
   ✓ Died: {died_pred_proba:.1%} probability

WHY INTERPRETABILITY MATTERS:
  • Build trust with users
  • Debug model behavior
  • Ensure fairness
  • Regulatory compliance
  • Improve model

NOTE: For advanced interpretability, install SHAP:
  brew install llvm
  pip install shap

NEXT: Build ML pipelines!
""")

print("\n✅ Model interpretability complete!")