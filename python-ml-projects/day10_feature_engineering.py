"""
Day 10: Feature Engineering Mastery
Advanced techniques for creating powerful features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   PolynomialFeatures, LabelEncoder)
from sklearn.feature_selection import (SelectKBest, f_classif, RFE, 
                                       mutual_info_classif)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FEATURE ENGINEERING MASTERY")
print("=" * 60)

# ============================================
# WHY FEATURE ENGINEERING MATTERS
# ============================================

print("\n1. WHY FEATURE ENGINEERING IS CRITICAL")
print("-" * 60)

print("""
"Feature engineering is the key to machine learning success.
Better features > Better algorithms" - Andrew Ng

IMPACT OF GOOD FEATURES:
  • Can improve accuracy by 10-30%
  • Often MORE important than algorithm choice
  • Encodes domain knowledge
  • Reduces model complexity

TODAY'S TECHNIQUES:
  1. Creating new features (combinations, ratios)
  2. Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
  3. Feature encoding (one-hot, label, target)
  4. Feature selection (filter, wrapper, embedded methods)
  5. Polynomial features
  6. Binning and discretization
  7. Date/time features
  8. Text features
""")

# ============================================
# LOAD TITANIC DATA
# ============================================

print("\n2. LOADING TITANIC DATASET")
print("-" * 60)

df = pd.read_csv('data/titanic.csv')
print(f"✅ Loaded {len(df)} passengers")
print(f"\nOriginal features: {df.columns.tolist()}")

# ============================================
# TECHNIQUE 1: CREATING NEW FEATURES
# ============================================

print("\n" + "=" * 60)
print("3. CREATING NEW FEATURES")
print("=" * 60)

df_engineered = df.copy()

# Fill missing values first
df_engineered['Age'].fillna(df_engineered['Age'].median(), inplace=True)
df_engineered['Embarked'].fillna(df_engineered['Embarked'].mode()[0], inplace=True)
df_engineered['Fare'].fillna(df_engineered['Fare'].median(), inplace=True)

print("\n3.1 Family Size Features")
print("-" * 40)

# Basic family size
df_engineered['FamilySize'] = df_engineered['SibSp'] + df_engineered['Parch'] + 1
print(f"Created: FamilySize (range: {df_engineered['FamilySize'].min()}-{df_engineered['FamilySize'].max()})")

# Is alone
df_engineered['IsAlone'] = (df_engineered['FamilySize'] == 1).astype(int)
print(f"Created: IsAlone ({df_engineered['IsAlone'].sum()} passengers alone)")

# Family category
df_engineered['FamilyCategory'] = pd.cut(df_engineered['FamilySize'], 
                                         bins=[0, 1, 4, 12],
                                         labels=['Alone', 'Small', 'Large'])
print(f"Created: FamilyCategory (Alone/Small/Large)")

print("\n3.2 Title Extraction")
print("-" * 40)

# Extract title from name
df_engineered['Title'] = df_engineered['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
print(f"Extracted {df_engineered['Title'].nunique()} unique titles")
print(f"Titles: {df_engineered['Title'].value_counts().head().to_dict()}")

# Group rare titles
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
    'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
    'Capt': 'Rare', 'Ms': 'Miss'
}
df_engineered['Title'] = df_engineered['Title'].map(title_mapping).fillna('Rare')
print(f"Simplified to: {df_engineered['Title'].unique()}")

print("\n3.3 Age Groups")
print("-" * 40)

df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], 
                                   bins=[0, 5, 12, 18, 35, 60, 100],
                                   labels=['Infant', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
print(f"Created age groups: {df_engineered['AgeGroup'].value_counts().to_dict()}")

# Age * Class interaction
df_engineered['Age_Class'] = df_engineered['Age'] * df_engineered['Pclass']
print(f"Created: Age_Class (interaction feature)")

print("\n3.4 Fare Features")
print("-" * 40)

# Fare per person
df_engineered['Fare_Per_Person'] = df_engineered['Fare'] / df_engineered['FamilySize']
print(f"Created: Fare_Per_Person (avg: {df_engineered['Fare_Per_Person'].mean():.2f})")

# Fare groups
df_engineered['FareGroup'] = pd.qcut(df_engineered['Fare'].rank(method='first'), 
                                     q=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
print(f"Created: FareGroup (quintiles)")

print("\n3.5 Cabin Features")
print("-" * 40)

# Deck from cabin
df_engineered['Deck'] = df_engineered['Cabin'].str[0]
df_engineered['Deck'].fillna('Unknown', inplace=True)

# Has cabin
df_engineered['HasCabin'] = df_engineered['Cabin'].notna().astype(int)
print(f"Created: Deck and HasCabin")
print(f"  Passengers with cabin: {df_engineered['HasCabin'].sum()}")

print("\n3.6 Interaction Features")
print("-" * 40)

# Sex + Class
df_engineered['Sex_Pclass'] = df_engineered['Sex'] + '_' + df_engineered['Pclass'].astype(str)
print(f"Created: Sex_Pclass combinations")

# Title + Age
df_engineered['Title_Age'] = df_engineered['Title'] + '_' + df_engineered['AgeGroup'].astype(str)
print(f"Created: Title_Age combinations")

print(f"\n✅ Feature engineering complete!")
print(f"   Original features: {len(df.columns)}")
print(f"   Engineered features: {len(df_engineered.columns)}")
print(f"   New features created: {len(df_engineered.columns) - len(df.columns)}")

# ============================================
# TECHNIQUE 2: FEATURE SCALING
# ============================================

print("\n" + "=" * 60)
print("4. FEATURE SCALING COMPARISON")
print("=" * 60)

# Prepare numeric features
numeric_features = ['Age', 'Fare', 'FamilySize', 'Fare_Per_Person', 'Age_Class']
X_numeric = df_engineered[numeric_features].copy()

print("\nOriginal statistics:")
print(X_numeric.describe())

# StandardScaler (z-score normalization)
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X_numeric)

print("\n4.1 StandardScaler (mean=0, std=1)")
print(f"  Mean after scaling: {X_standard.mean(axis=0).round(2)}")
print(f"  Std after scaling: {X_standard.std(axis=0).round(2)}")

# MinMaxScaler (0-1 range)
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_numeric)

print("\n4.2 MinMaxScaler (range 0-1)")
print(f"  Min after scaling: {X_minmax.min(axis=0).round(2)}")
print(f"  Max after scaling: {X_minmax.max(axis=0).round(2)}")

# RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_numeric)

print("\n4.3 RobustScaler (robust to outliers)")
print(f"  Uses median and IQR instead of mean and std")
print(f"  Median after scaling: {np.median(X_robust, axis=0).round(2)}")

print("""
WHEN TO USE EACH:
  • StandardScaler: Most common, works with most algorithms
  • MinMaxScaler: When you need bounded range (e.g., neural networks)
  • RobustScaler: When data has outliers
""")

# ============================================
# TECHNIQUE 3: ENCODING CATEGORICAL VARIABLES
# ============================================

print("\n" + "=" * 60)
print("5. ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

print("\n5.1 One-Hot Encoding (create dummy variables)")
print("-" * 40)

# One-hot encoding
df_encoded = pd.get_dummies(df_engineered[['Sex', 'Embarked', 'Title']], 
                           drop_first=True, prefix=['Sex', 'Emb', 'Title'])
print(f"Original columns: 3")
print(f"After one-hot: {df_encoded.shape[1]} columns")
print(f"New columns: {df_encoded.columns.tolist()}")

print("\n5.2 Label Encoding (ordinal variables)")
print("-" * 40)

# Label encoding for ordinal features
le = LabelEncoder()
df_engineered['AgeGroup_Encoded'] = le.fit_transform(df_engineered['AgeGroup'])
print(f"AgeGroup encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")

print("\n5.3 Target Encoding (advanced)")
print("-" * 40)

# Calculate survival rate by title
target_encoding = df_engineered.groupby('Title')['Survived'].mean()
df_engineered['Title_Target_Encoded'] = df_engineered['Title'].map(target_encoding)
print(f"Survival rate by title:")
print(target_encoding.round(3))

# ============================================
# TECHNIQUE 4: POLYNOMIAL FEATURES
# ============================================

print("\n" + "=" * 60)
print("6. POLYNOMIAL FEATURES")
print("=" * 60)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_sample = df_engineered[['Age', 'Fare']].values[:5]

X_poly = poly.fit_transform(X_sample)

print(f"Original features: {['Age', 'Fare']}")
print(f"Polynomial features (degree=2): {poly.get_feature_names_out(['Age', 'Fare'])}")
print(f"\nExample transformation:")
print(f"  Original: Age={X_sample[0,0]:.1f}, Fare={X_sample[0,1]:.1f}")
print(f"  Polynomial: {X_poly[0]}")
print(f"  Features: Age, Fare, Age², Age×Fare, Fare²")

# ============================================
# TECHNIQUE 5: FEATURE SELECTION
# ============================================

print("\n" + "=" * 60)
print("7. FEATURE SELECTION METHODS")
print("=" * 60)

# Prepare full feature set
features_to_use = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                   'FamilySize', 'IsAlone', 'Title', 'HasCabin', 
                   'Fare_Per_Person', 'Age_Class']

X_full = pd.get_dummies(df_engineered[features_to_use], drop_first=True)
y = df_engineered['Survived']

print(f"\nTotal features available: {X_full.shape[1]}")

print("\n7.1 Filter Method: SelectKBest (univariate)")
print("-" * 40)

# SelectKBest
selector_kbest = SelectKBest(f_classif, k=10)
X_kbest = selector_kbest.fit_transform(X_full, y)

# Get selected features
selected_mask = selector_kbest.get_support()
selected_features = X_full.columns[selected_mask].tolist()

print(f"Selected top 10 features:")
feature_scores = pd.DataFrame({
    'Feature': X_full.columns,
    'Score': selector_kbest.scores_
}).sort_values('Score', ascending=False).head(10)
print(feature_scores.to_string(index=False))

print("\n7.2 Wrapper Method: Recursive Feature Elimination (RFE)")
print("-" * 40)

# RFE with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rf, n_features_to_select=10)
X_rfe = rfe.fit_transform(X_full, y)

rfe_features = X_full.columns[rfe.support_].tolist()
print(f"RFE selected features: {rfe_features}")

print("\n7.3 Embedded Method: Feature Importance from Random Forest")
print("-" * 40)

# Train RF and get importance
rf.fit(X_full, y)
feature_importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print(f"Top 10 by importance:")
print(feature_importance.to_string(index=False))

# ============================================
# COMPARING FEATURE SETS
# ============================================

print("\n" + "=" * 60)
print("8. COMPARING FEATURE SETS")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# Test different feature sets
results = []

# All features
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
scores_all = cross_val_score(rf_all, X_train, y_train, cv=5)
results.append({
    'Method': 'All Features',
    'Num Features': X_full.shape[1],
    'CV Score': scores_all.mean(),
    'CV Std': scores_all.std()
})
print(f"\n1. All Features ({X_full.shape[1]}): {scores_all.mean():.4f} ± {scores_all.std():.4f}")

# SelectKBest features
X_train_kb = X_train.iloc[:, selected_mask]
X_test_kb = X_test.iloc[:, selected_mask]
rf_kb = RandomForestClassifier(n_estimators=100, random_state=42)
scores_kb = cross_val_score(rf_kb, X_train_kb, y_train, cv=5)
results.append({
    'Method': 'SelectKBest',
    'Num Features': X_train_kb.shape[1],
    'CV Score': scores_kb.mean(),
    'CV Std': scores_kb.std()
})
print(f"2. SelectKBest (10): {scores_kb.mean():.4f} ± {scores_kb.std():.4f}")

# RFE features
X_train_rfe = X_train.iloc[:, rfe.support_]
X_test_rfe = X_test.iloc[:, rfe.support_]
rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rfe = cross_val_score(rf_rfe, X_train_rfe, y_train, cv=5)
results.append({
    'Method': 'RFE',
    'Num Features': X_train_rfe.shape[1],
    'CV Score': scores_rfe.mean(),
    'CV Std': scores_rfe.std()
})
print(f"3. RFE (10): {scores_rfe.mean():.4f} ± {scores_rfe.std():.4f}")

# Top importance features
top_10_features = feature_importance.head(10)['Feature'].tolist()
X_train_imp = X_train[top_10_features]
X_test_imp = X_test[top_10_features]
rf_imp = RandomForestClassifier(n_estimators=100, random_state=42)
scores_imp = cross_val_score(rf_imp, X_train_imp, y_train, cv=5)
results.append({
    'Method': 'Top Importance',
    'Num Features': len(top_10_features),
    'CV Score': scores_imp.mean(),
    'CV Std': scores_imp.std()
})
print(f"4. Top Importance (10): {scores_imp.mean():.4f} ± {scores_imp.std():.4f}")

results_df = pd.DataFrame(results)
best_method = results_df.loc[results_df['CV Score'].idxmax()]

print(f"\n🏆 BEST METHOD: {best_method['Method']}")
print(f"   CV Score: {best_method['CV Score']:.4f}")
print(f"   Features: {int(best_method['Num Features'])}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("9. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Engineering Analysis', fontsize=18, fontweight='bold')

# Plot 1: Feature Importance
ax1 = axes[0, 0]
top_15 = feature_importance.head(15)
bars = ax1.barh(range(len(top_15)), top_15['Importance'],
               color='skyblue', edgecolor='black')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['Feature'], fontsize=9)
ax1.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax1.set_title('Top 15 Features by Importance', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Plot 2: Feature Selection Comparison
ax2 = axes[0, 1]
methods = results_df['Method'].values
scores = results_df['CV Score'].values
errors = results_df['CV Std'].values

bars = ax2.bar(range(len(methods)), scores, yerr=errors,
              capsize=5, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
ax2.set_ylabel('Cross-Validation Score', fontsize=11, fontweight='bold')
ax2.set_title('Feature Selection Methods Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim(0.75, 0.85)
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Correlation Heatmap (top features)
ax3 = axes[1, 0]
top_features_for_corr = top_10_features[:8]  # Top 8 for readability
corr_matrix = X_full[top_features_for_corr].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')

# Plot 4: Feature Count Impact
ax4 = axes[1, 1]
num_features = results_df['Num Features'].values

ax4.scatter(num_features, scores, s=300, alpha=0.7, edgecolors='black', linewidth=2)

for i, method in enumerate(methods):
    ax4.annotate(method, (num_features[i], scores[i]),
                fontsize=9, ha='center', va='bottom')

ax4.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cross-Validation Score', fontsize=11, fontweight='bold')
ax4.set_title('Features vs Performance', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/43_feature_engineering_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/43_feature_engineering_analysis.png")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: FEATURE ENGINEERING")
print("=" * 60)

print(f"""
TECHNIQUES MASTERED:

1. FEATURE CREATION:
   ✓ Domain knowledge features (FamilySize, Title)
   ✓ Interaction features (Age × Class)
   ✓ Ratio features (Fare per person)
   ✓ Binning (Age groups, Fare groups)
   
2. FEATURE SCALING:
   ✓ StandardScaler: Most common (mean=0, std=1)
   ✓ MinMaxScaler: For bounded ranges (0-1)
   ✓ RobustScaler: For data with outliers
   
3. FEATURE ENCODING:
   ✓ One-hot encoding: For nominal categories
   ✓ Label encoding: For ordinal categories
   ✓ Target encoding: Using target statistics
   
4. POLYNOMIAL FEATURES:
   ✓ Create interactions automatically
   ✓ Capture non-linear relationships
   ⚠️  Watch out for overfitting
   
5. FEATURE SELECTION:
   ✓ Filter methods: Fast, univariate (SelectKBest)
   ✓ Wrapper methods: Slow, optimal (RFE)
   ✓ Embedded methods: Built-in (RF importance)

RESULTS:
  • Created {len(df_engineered.columns) - len(df.columns)} new features
  • Best method: {best_method['Method']}
  • Best score: {best_method['CV Score']:.4f}
  • Optimal features: {int(best_method['Num Features'])}

GOLDEN RULES:
  1. More features ≠ Better performance
  2. Feature engineering > Algorithm selection
  3. Domain knowledge is crucial
  4. Always validate with cross-validation
  5. Remove correlated features
  6. Scale features for distance-based algorithms

NEXT: Model interpretability with SHAP!
""")

print("\n✅ Feature engineering mastery complete!")