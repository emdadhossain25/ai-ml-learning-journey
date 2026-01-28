"""
Day 19: Kaggle Titanic - First Submission (Baseline)
Quick and simple model to get on the leaderboard!
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("=" * 60)
print("KAGGLE TITANIC - FIRST SUBMISSION")
print("=" * 60)

# Load data (update path to where you downloaded)
train = pd.read_csv('/Users/emdadhossain/Downloads/train.csv')  # Update this path!
test = pd.read_csv('/Users/emdadhossain/Downloads/test.csv')    # Update this path!

print(f"\n✅ Loaded data:")
print(f"   Training: {len(train)} passengers")
print(f"   Test: {len(test)} passengers")

print("\nTraining data columns:")
print(train.columns.tolist())

print("\nFirst few rows:")
print(train.head())

print("\nSurvival rate in training:")
print(f"   Survived: {train['Survived'].sum()} ({train['Survived'].mean():.1%})")
print(f"   Died: {len(train) - train['Survived'].sum()} ({(1-train['Survived'].mean()):.1%})")

# ============================================
# SIMPLE FEATURE ENGINEERING
# ============================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

def prepare_features(df):
    """Simple feature engineering"""
    df = df.copy()
    
    # Fill missing ages with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fill missing fare with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is alone?
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                        'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                        'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Age bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 120], 
                          labels=['Child', 'Teen', 'Adult', 'Senior'])
    
    # Fare bins
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'High', 'VeryHigh'])
    
    return df

train_processed = prepare_features(train)
test_processed = prepare_features(test)

print("✅ Features engineered:")
print("   - Filled missing values")
print("   - Created FamilySize")
print("   - Created IsAlone")
print("   - Extracted Title from Name")
print("   - Created AgeBin")
print("   - Created FareBin")

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================

print("\n" + "=" * 60)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 60)

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']

# Encode categorical
le = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']:
    train_processed[col] = le.fit_transform(train_processed[col].astype(str))
    test_processed[col] = le.transform(test_processed[col].astype(str))

X_train = train_processed[features]
y_train = train_processed['Survived']
X_test = test_processed[features]

print(f"✅ Training features shape: {X_train.shape}")
print(f"✅ Test features shape: {X_test.shape}")

# ============================================
# TRAIN MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Training accuracy
train_accuracy = model.score(X_train, y_train)
print(f"✅ Model trained!")
print(f"   Training accuracy: {train_accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# ============================================
# MAKE PREDICTIONS
# ============================================

print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

predictions = model.predict(X_test)

print(f"✅ Predictions made for {len(predictions)} passengers")
print(f"   Predicted survived: {predictions.sum()} ({predictions.mean():.1%})")
print(f"   Predicted died: {len(predictions) - predictions.sum()} ({(1-predictions.mean()):.1%})")

# ============================================
# CREATE SUBMISSION FILE
# ============================================

print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('titanic_submission.csv', index=False)

print("✅ Submission file created: titanic_submission.csv")
print(f"   {len(submission)} predictions")
print("\nFirst few predictions:")
print(submission.head(10))

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("""
1. Go to Kaggle Titanic competition
2. Click "Submit Predictions"
3. Upload: titanic_submission.csv
4. Submit and see your score!
5. Check leaderboard position

EXPECTED SCORE: ~77-78% accuracy (decent baseline!)

TO IMPROVE:
- Better feature engineering
- Hyperparameter tuning
- Ensemble methods
- Learn from top notebooks
""")

print("\n🎉 CONGRATULATIONS! Your first Kaggle submission ready!")
print("=" * 60)
