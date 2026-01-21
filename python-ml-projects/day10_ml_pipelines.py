"""
Day 10: ML Pipelines & Automation
Building production-ready, reproducible workflows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ML PIPELINES & AUTOMATION")
print("=" * 60)

# ============================================
# WHY PIPELINES MATTER
# ============================================

print("\n1. WHY USE ML PIPELINES?")
print("-" * 60)

print("""
WITHOUT PIPELINES (Manual Process):
  1. Load data
  2. Fill missing values
  3. Scale features
  4. Encode categories
  5. Train model
  6. ...repeat for test data (ERROR PRONE!)

WITH PIPELINES:
  • All steps automated
  • Reproducible workflow
  • Prevents data leakage
  • Easy to deploy
  • Grid search over entire pipeline

BENEFITS:
  ✓ Cleaner code
  ✓ Less error-prone
  ✓ Production-ready
  ✓ Easier experimentation
  ✓ Prevents train/test contamination
""")

# ============================================
# LOAD DATA
# ============================================

print("\n2. LOADING RAW DATA")
print("-" * 60)

df = pd.read_csv('data/titanic.csv')

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data loaded: {len(X)} passengers")
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
print(f"\nOriginal columns: {X.columns.tolist()}")
print(f"\nMissing values:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

# ============================================
# CUSTOM TRANSFORMERS
# ============================================

print("\n" + "=" * 60)
print("3. CREATING CUSTOM TRANSFORMERS")
print("=" * 60)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Family size
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        
        # Title extraction
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
        X['Title'] = X['Title'].map(title_map).fillna('Rare')
        
        # Fare per person
        X['Fare_Per_Person'] = X['Fare'] / X['FamilySize']
        
        # Has cabin
        X['HasCabin'] = X['Cabin'].notna().astype(int)
        
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific columns"""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

print("✅ Custom transformers created:")
print("   • FeatureEngineer: Creates new features")
print("   • FeatureSelector: Selects columns")

# ============================================
# BUILDING PIPELINE - SIMPLE VERSION
# ============================================

print("\n" + "=" * 60)
print("4. SIMPLE PIPELINE (Numeric Only)")
print("=" * 60)

# Define numeric features
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']

# Simple pipeline for numeric data
simple_pipeline = Pipeline([
    ('selector', FeatureSelector(numeric_features)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Pipeline steps:")
for i, (name, step) in enumerate(simple_pipeline.steps, 1):
    print(f"  {i}. {name}: {type(step).__name__}")

# Train pipeline
print("\nTraining simple pipeline...")
simple_pipeline.fit(X_train, y_train)

# Evaluate
simple_score = simple_pipeline.score(X_test, y_test)
print(f"✅ Simple Pipeline Accuracy: {simple_score:.4f}")

# ============================================
# ADVANCED PIPELINE - FULL PREPROCESSING
# ============================================

print("\n" + "=" * 60)
print("5. ADVANCED PIPELINE (Full Preprocessing)")
print("=" * 60)

# Define feature types
numeric_features = ['Age', 'Fare', 'FamilySize', 'Fare_Per_Person']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']

# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Complete pipeline
advanced_pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Advanced Pipeline Architecture:")
print("\n1. Feature Engineering:")
print("   • Creates: FamilySize, IsAlone, Title, Fare_Per_Person, HasCabin")

print("\n2. Preprocessing (ColumnTransformer):")
print("   Numeric Pipeline:")
for i, (name, step) in enumerate(numeric_transformer.steps, 1):
    print(f"     {i}. {name}: {type(step).__name__}")

print("   Categorical Pipeline:")
for i, (name, step) in enumerate(categorical_transformer.steps, 1):
    print(f"     {i}. {name}: {type(step).__name__}")

print("\n3. Model:")
print("   • RandomForestClassifier")

# Train pipeline
print("\nTraining advanced pipeline...")
advanced_pipeline.fit(X_train, y_train)

# Evaluate
advanced_score = advanced_pipeline.score(X_test, y_test)
print(f"✅ Advanced Pipeline Accuracy: {advanced_score:.4f}")

print(f"\nImprovement: +{(advanced_score - simple_score)*100:.2f}%")

# ============================================
# CROSS-VALIDATION WITH PIPELINE
# ============================================

print("\n" + "=" * 60)
print("6. CROSS-VALIDATION WITH PIPELINE")
print("=" * 60)

print("Running 5-fold cross-validation...")
cv_scores = cross_val_score(advanced_pipeline, X_train, y_train, cv=5)

print(f"\nCross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean CV Score: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# ============================================
# GRID SEARCH WITH PIPELINE
# ============================================

print("\n" + "=" * 60)
print("7. GRID SEARCH OVER ENTIRE PIPELINE")
print("=" * 60)

print("Searching hyperparameters across pipeline...")

# Define parameter grid
# Note: Use 'step_name__parameter' syntax
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")

# Grid search
grid_search = GridSearchCV(
    advanced_pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

print("\nRunning grid search...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Grid search complete!")
print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# ============================================
# PIPELINE PERSISTENCE
# ============================================

print("\n" + "=" * 60)
print("8. SAVING PIPELINE FOR PRODUCTION")
print("=" * 60)

# Save the best pipeline
best_pipeline = grid_search.best_estimator_

# Save to file
model_filename = 'models/titanic_pipeline.pkl'
joblib.dump(best_pipeline, model_filename)
print(f"✅ Pipeline saved to: {model_filename}")

# Load and test
loaded_pipeline = joblib.load(model_filename)
loaded_score = loaded_pipeline.score(X_test, y_test)
print(f"✅ Pipeline loaded successfully")
print(f"   Test accuracy: {loaded_score:.4f}")

# Make predictions with loaded pipeline
print("\nTesting loaded pipeline on new data...")

# Create sample passenger
new_passenger = pd.DataFrame({
    'Pclass': [1],
    'Name': ['Miss. Jane Smith'],
    'Sex': ['female'],
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [100],
    'Embarked': ['C'],
    'Cabin': ['C85'],
    'Ticket': ['12345']
})

prediction = loaded_pipeline.predict(new_passenger)
probability = loaded_pipeline.predict_proba(new_passenger)[0, 1]

print(f"\nNew passenger prediction:")
print(f"  Survived: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"  Probability: {probability:.2%}")

# ============================================
# COMPARISON OF APPROACHES
# ============================================

print("\n" + "=" * 60)
print("9. COMPARISON: MANUAL vs PIPELINE")
print("=" * 60)

comparison = pd.DataFrame({
    'Approach': ['Manual Process', 'Simple Pipeline', 'Advanced Pipeline', 'Optimized Pipeline'],
    'Accuracy': [0.78, simple_score, advanced_score, grid_search.score(X_test, y_test)],
    'Code Lines': [100, 20, 30, 35],
    'Reproducible': ['No', 'Yes', 'Yes', 'Yes'],
    'Production Ready': ['No', 'Partially', 'Yes', 'Yes']
})

print("\n" + comparison.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("10. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ML Pipelines Analysis', fontsize=18, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
approaches = comparison['Approach'].values
accuracies = comparison['Accuracy'].values
colors_list = ['lightgray', 'skyblue', 'lightgreen', 'gold']

bars = ax1.barh(approaches, accuracies, color=colors_list, 
               edgecolor='black', linewidth=2)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Approach Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0.75, 0.88)
ax1.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, accuracies):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=11, fontweight='bold')

# Plot 2: Pipeline Flow Diagram
ax2 = axes[0, 1]
ax2.axis('off')

pipeline_diagram = """
┌─────────────────────────────────────────────┐
│        ADVANCED PIPELINE FLOW               │
├─────────────────────────────────────────────┤
│                                             │
│  1. RAW DATA                                │
│     ↓                                       │
│  2. FEATURE ENGINEERING                     │
│     • FamilySize = SibSp + Parch + 1        │
│     • Title from Name                       │
│     • Fare per person                       │
│     ↓                                       │
│  3. PREPROCESSING                           │
│     ┌─────────────┬──────────────┐          │
│     │  Numeric    │  Categorical │          │
│     ├─────────────┼──────────────┤          │
│     │ • Impute    │ • Impute     │          │
│     │ • Scale     │ • One-Hot    │          │
│     └─────────────┴──────────────┘          │
│     ↓                                       │
│  4. MACHINE LEARNING MODEL                  │
│     • Random Forest Classifier              │
│     ↓                                       │
│  5. PREDICTION                              │
│     • Survived: Yes/No                      │
│     • Probability: 0-1                      │
│                                             │
│  ADVANTAGES:                                │
│  ✓ Automated                                │
│  ✓ Reproducible                             │
│  ✓ No data leakage                          │
│  ✓ Production-ready                         │
│                                             │
└─────────────────────────────────────────────┘
"""

ax2.text(0.1, 0.5, pipeline_diagram, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Plot 3: Grid Search Results
ax3 = axes[1, 0]

# Get top 10 configurations
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.nsmallest(10, 'rank_test_score')

ax3.barh(range(len(top_10)), top_10['mean_test_score'],
        xerr=top_10['std_test_score'], capsize=3,
        alpha=0.7, color='lightgreen', edgecolor='black')
ax3.set_yticks(range(len(top_10)))
ax3.set_yticklabels([f"Config {i+1}" for i in range(len(top_10))], fontsize=9)
ax3.set_xlabel('Cross-Validation Score', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Configurations (Grid Search)', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# Plot 4: Cross-Validation Scores
ax4 = axes[1, 1]

cv_results = {
    'Simple Pipeline': cross_val_score(simple_pipeline, X_train, y_train, cv=5),
    'Advanced Pipeline': cross_val_score(advanced_pipeline, X_train, y_train, cv=5),
    'Optimized Pipeline': cross_val_score(best_pipeline, X_train, y_train, cv=5)
}

positions = []
data = []
labels = []

for i, (name, scores) in enumerate(cv_results.items()):
    positions.extend([i+1]*len(scores))
    data.extend(scores)
    labels.append(name)

ax4.scatter(positions, data, s=100, alpha=0.6, edgecolors='black', linewidth=1)

# Add mean lines
for i, (name, scores) in enumerate(cv_results.items(), 1):
    mean_score = scores.mean()
    ax4.plot([i-0.2, i+0.2], [mean_score, mean_score], 
            'r-', linewidth=3, label='Mean' if i == 1 else '')

ax4.set_xticks(range(1, len(labels)+1))
ax4.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
ax4.set_ylabel('Cross-Validation Score', fontsize=12, fontweight='bold')
ax4.set_title('Cross-Validation Stability', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('plots/45_ml_pipelines.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/45_ml_pipelines.png")

# ============================================
# PIPELINE CODE TEMPLATE
# ============================================

print("\n" + "=" * 60)
print("11. PRODUCTION PIPELINE TEMPLATE")
print("=" * 60)

template = '''
# PRODUCTION ML PIPELINE TEMPLATE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Define pipeline
pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', ColumnTransformer(...)),
    ('model', RandomForestClassifier())
])

# 2. Train
pipeline.fit(X_train, y_train)

# 3. Evaluate
score = pipeline.score(X_test, y_test)

# 4. Save
joblib.dump(pipeline, 'model.pkl')

# 5. Deploy
loaded_pipeline = joblib.load('model.pkl')
prediction = loaded_pipeline.predict(new_data)
'''

print(template)

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: ML PIPELINES")
print("=" * 60)

print(f"""
WHAT WE BUILT:
  ✓ Custom transformers (FeatureEngineer, FeatureSelector)
  ✓ Simple pipeline (numeric only)
  ✓ Advanced pipeline (full preprocessing)
  ✓ Grid search over pipeline
  ✓ Saved production-ready model

RESULTS:
  • Simple Pipeline: {simple_score:.4f}
  • Advanced Pipeline: {advanced_score:.4f}
  • Optimized Pipeline: {grid_search.score(X_test, y_test):.4f}
  • Improvement: +{(grid_search.score(X_test, y_test) - simple_score)*100:.2f}%

PIPELINE BENEFITS:
  1. AUTOMATION: All steps in one object
  2. REPRODUCIBILITY: Same process every time
  3. NO DATA LEAKAGE: Test data never seen during fit
  4. PRODUCTION READY: Save/load easily
  5. GRID SEARCH: Optimize entire workflow
  6. CLEANER CODE: Less error-prone

KEY COMPONENTS:
  • Pipeline: Sequential steps
  • ColumnTransformer: Different transformations per column type
  • Custom Transformers: Domain-specific logic
  • joblib: Model persistence

PRODUCTION WORKFLOW:
  1. Build pipeline
  2. Grid search for best parameters
  3. Train on full data
  4. Save with joblib
  5. Load in production
  6. Make predictions

NEXT: Deploy as REST API!
""")

print("\n✅ ML Pipelines mastery complete!")