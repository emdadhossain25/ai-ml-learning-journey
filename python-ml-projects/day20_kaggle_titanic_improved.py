"""
Day 20: Kaggle Titanic - Improved Model
Advanced feature engineering + ensemble methods
Goal: Beat yesterday's baseline by 2-3%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("KAGGLE TITANIC - IMPROVED MODEL (DAY 20)")
print("=" * 60)

# ============================================
# LOAD DATA
# ============================================

print("\n1. LOADING DATA")
print("-" * 60)

# Update paths to where you downloaded Kaggle data
train = pd.read_csv('/Users/emdadhossain/Downloads/train.csv')
test = pd.read_csv('/Users/emdadhossain/Downloads/test.csv')

print(f"✅ Training: {len(train)} passengers")
print(f"✅ Test: {len(test)} passengers")

# Combine for consistent preprocessing
full_data = pd.concat([train, test], ignore_index=True, sort=False)
print(f"✅ Combined: {len(full_data)} passengers")

# ============================================
# ADVANCED FEATURE ENGINEERING
# ============================================

print("\n" + "=" * 60)
print("2. ADVANCED FEATURE ENGINEERING")
print("=" * 60)

def advanced_features(df):
    """Create advanced features"""
    df = df.copy()
    
    # ============================================
    # FEATURE 1: Title from Name (More Detailed)
    # ============================================
    print("\n📝 Extracting Title from Name...")
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                        'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                        'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    print(f"   Titles found: {df['Title'].unique()}")
    print(f"   Title counts: {df['Title'].value_counts().to_dict()}")
    
    # ============================================
    # FEATURE 2: Family Size & Type
    # ============================================
    print("\n👨‍👩‍👧‍👦 Creating Family Features...")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Categorize family size
    df['FamilyType'] = 'Medium'
    df.loc[df['FamilySize'] == 1, 'FamilyType'] = 'Alone'
    df.loc[df['FamilySize'] >= 5, 'FamilyType'] = 'Large'
    
    print(f"   Family types: {df['FamilyType'].value_counts().to_dict()}")
    
    # ============================================
    # FEATURE 3: Cabin Letter (Deck)
    # ============================================
    print("\n🚢 Extracting Deck from Cabin...")
    df['Deck'] = df['Cabin'].str[0]  # First letter
    df['Deck'].fillna('Unknown', inplace=True)
    
    print(f"   Decks: {df['Deck'].unique()}")
    print(f"   Unknown cabins: {(df['Deck'] == 'Unknown').sum()}")
    
    # ============================================
    # FEATURE 4: Ticket Prefix
    # ============================================
    print("\n🎫 Extracting Ticket Prefix...")
    df['TicketPrefix'] = df['Ticket'].str.split().str[0]
    df['TicketPrefix'] = df['TicketPrefix'].str.replace('.', '')
    df['TicketPrefix'] = df['TicketPrefix'].str.replace('/', '')
    
    # Group rare prefixes
    prefix_counts = df['TicketPrefix'].value_counts()
    rare_prefixes = prefix_counts[prefix_counts < 10].index
    df['TicketPrefix'] = df['TicketPrefix'].replace(rare_prefixes, 'Rare')
    
    print(f"   Unique prefixes: {df['TicketPrefix'].nunique()}")
    
    # ============================================
    # FEATURE 5: Name Length
    # ============================================
    print("\n📏 Creating Name Length...")
    df['NameLength'] = df['Name'].str.len()
    
    # Bin name length
    df['NameLengthBin'] = pd.cut(df['NameLength'], 
                                   bins=[0, 20, 30, 40, 100],
                                   labels=['Short', 'Medium', 'Long', 'VeryLong'])
    
    print(f"   Name length range: {df['NameLength'].min()}-{df['NameLength'].max()}")
    
    # ============================================
    # FEATURE 6: Age Filling (More Sophisticated)
    # ============================================
    print("\n👤 Filling Missing Ages...")
    
    # Fill age based on Title median
    age_by_title = df.groupby('Title')['Age'].median()
    for title in df['Title'].unique():
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_by_title[title]
    
    # Fill remaining with overall median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    print(f"   Ages filled. Missing: {df['Age'].isnull().sum()}")
    
    # ============================================
    # FEATURE 7: Age Groups (More Granular)
    # ============================================
    print("\n🎂 Creating Age Groups...")
    df['AgeGroup'] = pd.cut(df['Age'], 
                             bins=[0, 4, 12, 18, 30, 50, 80],
                             labels=['Baby', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    
    print(f"   Age groups: {df['AgeGroup'].value_counts().to_dict()}")
    
    # ============================================
    # FEATURE 8: Fare Filling & Binning
    # ============================================
    print("\n💰 Processing Fares...")
    
    # Fill missing fare
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Fare per person (if sharing ticket)
    df['FarePerPerson'] = df['Fare'] / (df['FamilySize'])
    
    # Fare bins
    df['FareBin'] = pd.qcut(df['Fare'], q=5, duplicates='drop',
                             labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    print(f"   Fare range: ${df['Fare'].min():.2f} - ${df['Fare'].max():.2f}")
    
    # ============================================
    # FEATURE 9: Embarked
    # ============================================
    print("\n⚓ Processing Embarked...")
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    print(f"   Embarked: {df['Embarked'].value_counts().to_dict()}")
    
    # ============================================
    # FEATURE 10: Is Alone
    # ============================================
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # ============================================
    # FEATURE 11: Age * Class Interaction
    # ============================================
    print("\n🔗 Creating Interaction Features...")
    df['Age_Class'] = df['Age'] * df['Pclass']
    df['Fare_Class'] = df['Fare'] * df['Pclass']
    
    print("✅ All features created!")
    
    return df

# Apply feature engineering
full_data = advanced_features(full_data)

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================

print("\n" + "=" * 60)
print("3. ENCODING CATEGORICAL FEATURES")
print("=" * 60)

# Select features for modeling
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Embarked', 'Title', 'FamilySize', 'FamilyType',
                   'Deck', 'TicketPrefix', 'NameLength', 'NameLengthBin',
                   'AgeGroup', 'FarePerPerson', 'FareBin', 'IsAlone',
                   'Age_Class', 'Fare_Class']

# Label encode categorical features
le = LabelEncoder()
categorical_features = ['Sex', 'Embarked', 'Title', 'FamilyType', 'Deck', 
                        'TicketPrefix', 'NameLengthBin', 'AgeGroup', 'FareBin']

for col in categorical_features:
    full_data[col] = le.fit_transform(full_data[col].astype(str))

print(f"✅ Encoded {len(categorical_features)} categorical features")

# Split back into train and test
train_processed = full_data[:len(train)].copy()
test_processed = full_data[len(train):].copy()

X_train = train_processed[feature_columns]
y_train = train_processed['Survived']
X_test = test_processed[feature_columns]

print(f"\n✅ Final feature set:")
print(f"   Training shape: {X_train.shape}")
print(f"   Test shape: {X_test.shape}")
print(f"   Number of features: {len(feature_columns)}")

# ============================================
# MODEL TRAINING - MULTIPLE MODELS
# ============================================

print("\n" + "=" * 60)
print("4. TRAINING MULTIPLE MODELS")
print("=" * 60)

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Training accuracy
    train_acc = model.score(X_train, y_train)
    
    results.append({
        'Model': name,
        'CV Score': cv_mean,
        'CV Std': cv_std,
        'Train Acc': train_acc
    })
    
    print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"  Train Acc: {train_acc:.4f}")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# Select best model by CV score
best_model_name = results_df.loc[results_df['CV Score'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   CV Score: {results_df['CV Score'].max():.4f}")

# ============================================
# ENSEMBLE PREDICTION (Voting)
# ============================================

print("\n" + "=" * 60)
print("5. ENSEMBLE PREDICTION (VOTING)")
print("=" * 60)

print("Creating ensemble from all 3 models...")

# Get predictions from all models
predictions = []
for name, model in models.items():
    pred = model.predict(X_test)
    predictions.append(pred)

# Voting: majority wins
predictions = np.array(predictions).astype(int)
ensemble_pred = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(), 
    axis=0, 
    arr=predictions
)

print(f"✅ Ensemble predictions created")
print(f"   Predicted survived: {ensemble_pred.sum()} ({ensemble_pred.mean():.1%})")
print(f"   Predicted died: {len(ensemble_pred) - ensemble_pred.sum()}")

# ============================================
# FEATURE IMPORTANCE
# ============================================

print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = models['RandomForest']
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================
# CREATE SUBMISSION FILES
# ============================================

print("\n" + "=" * 60)
print("7. CREATING SUBMISSION FILES")
print("=" * 60)

# Submission 1: Best single model
best_pred = best_model.predict(X_test)
submission_best = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': best_pred
})
submission_best.to_csv('titanic_submission_day20_best.csv', index=False)
print(f"✅ Created: titanic_submission_day20_best.csv ({best_model_name})")

# Submission 2: Ensemble
submission_ensemble = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': ensemble_pred
})
submission_ensemble.to_csv('titanic_submission_day20_ensemble.csv', index=False)
print(f"✅ Created: titanic_submission_day20_ensemble.csv (Voting)")

print("\n" + "=" * 60)
print("COMPARISON WITH DAY 19")
print("=" * 60)

print("""
DAY 19 (Baseline):
- Features: 12 basic features
- Model: Single Random Forest
- Expected Score: ~77-78%

DAY 20 (Improved):
- Features: 20 advanced features
- Models: 3 models + ensemble voting
- New features:
  - Cabin deck extraction
  - Ticket prefix patterns
  - Name length
  - Age*Class interaction
  - Fare per person
  - More granular binning
- Expected Score: ~79-81% (+2-3% improvement!)

TRY BOTH SUBMISSIONS:
1. titanic_submission_day20_best.csv (single model)
2. titanic_submission_day20_ensemble.csv (ensemble)

See which scores higher on Kaggle leaderboard!
""")

print("\n" + "=" * 60)
print("NEXT STEPS TO IMPROVE FURTHER")
print("=" * 60)

print("""
1. HYPERPARAMETER TUNING
   • GridSearchCV for optimal parameters
   • Try different max_depth, n_estimators

2. MORE ADVANCED ENSEMBLES
   • Stacking (use predictions as features)
   • Weighted voting (give best model more weight)
   • Blending

3. FEATURE ENGINEERING V3
   • Cabin number extraction
   • Ticket number patterns
   • Port of embarkation + Class interaction
   • Surname extraction (family groups)

4. STUDY TOP SOLUTIONS
   • Read top 10 notebooks on Kaggle
   • Implement their best techniques
   • Combine ideas

TARGET: Top 30% (79-81% accuracy)
STRETCH: Top 20% (82-83% accuracy)
""")

print("\n🎉 IMPROVED MODEL READY!")
print("Upload both submissions to Kaggle and compare scores!")
print("=" * 60)