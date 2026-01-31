"""
Day 22: Kaggle House Prices - Advanced Regression
Predict house sale prices using regression techniques
First regression competition (vs Titanic classification)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("KAGGLE HOUSE PRICES - REGRESSION MODEL")
print("=" * 60)

# ============================================
# LOAD DATA
# ============================================

print("\n1. LOADING DATA")
print("-" * 60)

# Update paths to where you downloaded data
train = pd.read_csv('/Users/emdadhossain/Downloads/train.csv')
test = pd.read_csv('/Users/emdadhossain/Downloads/test.csv')

print(f"✅ Training data: {train.shape}")
print(f"✅ Test data: {test.shape}")

# Store test IDs for submission
test_ids = test['Id']

# Target variable
y_train = train['SalePrice']

print(f"\n📊 TARGET VARIABLE (SalePrice):")
print(f"   Mean: ${y_train.mean():,.0f}")
print(f"   Median: ${y_train.median():,.0f}")
print(f"   Min: ${y_train.min():,.0f}")
print(f"   Max: ${y_train.max():,.0f}")
print(f"   Std: ${y_train.std():,.0f}")

# Combine for consistent preprocessing
all_data = pd.concat([train.drop('SalePrice', axis=1), test], ignore_index=True)
print(f"\n✅ Combined dataset: {all_data.shape}")

# ============================================
# EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nColumns with missing values:")
missing = all_data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\n{missing.head(10)}")
print(f"\nTotal columns with missing data: {len(missing)}")

print("\nData types:")
print(all_data.dtypes.value_counts())

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

def feature_engineering(df):
    """Advanced feature engineering for house prices"""
    df = df.copy()
    
    print("\n🔧 Creating new features...")
    
    # ============================================
    # FEATURE 1: Total Square Footage
    # ============================================
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # ============================================
    # FEATURE 2: Total Bathrooms
    # ============================================
    df['TotalBath'] = (df['FullBath'] + 
                       (0.5 * df['HalfBath']) + 
                       df['BsmtFullBath'] + 
                       (0.5 * df['BsmtHalfBath']))
    
    # ============================================
    # FEATURE 3: Total Porch Area
    # ============================================
    df['TotalPorchSF'] = (df['OpenPorchSF'] + 
                          df['3SsnPorch'] + 
                          df['EnclosedPorch'] + 
                          df['ScreenPorch'] + 
                          df['WoodDeckSF'])
    
    # ============================================
    # FEATURE 4: House Age (at time of sale)
    # ============================================
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    # ============================================
    # FEATURE 5: Is Remodeled
    # ============================================
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    
    # ============================================
    # FEATURE 6: Has Pool
    # ============================================
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    # ============================================
    # FEATURE 7: Has Garage
    # ============================================
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    # ============================================
    # FEATURE 8: Has Basement
    # ============================================
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    
    # ============================================
    # FEATURE 9: Has Fireplace
    # ============================================
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    # ============================================
    # FEATURE 10: Quality Scores (multiply related features)
    # ============================================
    df['OverallQualCond'] = df['OverallQual'] * df['OverallCond']
    df['QualGrLiv'] = df['OverallQual'] * df['GrLivArea']
    df['QualBsmt'] = df['OverallQual'] * df['TotalBsmtSF']
    df['QualGarage'] = df['OverallQual'] * df['GarageArea']
    
    print(f"✅ Created 13+ new features")
    
    return df

all_data = feature_engineering(all_data)

# ============================================
# HANDLE MISSING VALUES
# ============================================

print("\n" + "=" * 60)
print("4. HANDLING MISSING VALUES")
print("=" * 60)

def handle_missing(df):
    """Handle missing values intelligently"""
    df = df.copy()
    
    print("\n🔧 Filling missing values...")
    
    # For categorical features where NA means "None"
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    for col in none_cols:
        if col in df.columns:
            df[col].fillna('None', inplace=True)
    
    # For numerical features where NA means 0
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    for col in zero_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # LotFrontage: fill with median by neighborhood
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # MSZoning: fill with mode
    if 'MSZoning' in df.columns:
        df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
    
    # Utilities: almost all 'AllPub', fill with mode
    if 'Utilities' in df.columns:
        df['Utilities'].fillna(df['Utilities'].mode()[0], inplace=True)
    
    # Functional: fill with 'Typ' (typical)
    if 'Functional' in df.columns:
        df['Functional'].fillna('Typ', inplace=True)
    
    # Electrical: fill with mode
    if 'Electrical' in df.columns:
        df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    
    # KitchenQual: fill with mode
    if 'KitchenQual' in df.columns:
        df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
    
    # Exterior: fill with mode
    if 'Exterior1st' in df.columns:
        df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
    if 'Exterior2nd' in df.columns:
        df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0], inplace=True)
    
    # SaleType: fill with mode
    if 'SaleType' in df.columns:
        df['SaleType'].fillna(df['SaleType'].mode()[0], inplace=True)
    
    print(f"✅ Missing values handled")
    print(f"   Remaining missing: {df.isnull().sum().sum()}")
    
    return df

all_data = handle_missing(all_data)

# ============================================
# ENCODE CATEGORICAL VARIABLES
# ============================================

print("\n" + "=" * 60)
print("5. ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

# Get categorical columns
categorical_cols = all_data.select_dtypes(include=['object']).columns
print(f"\n📊 Found {len(categorical_cols)} categorical columns")

# One-hot encoding for categorical variables
all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

print(f"✅ After encoding: {all_data.shape[1]} features")

# ============================================
# SPLIT BACK INTO TRAIN/TEST
# ============================================

print("\n" + "=" * 60)
print("6. PREPARING FINAL DATASETS")
print("=" * 60)

X_train = all_data[:len(train)]
X_test = all_data[len(train):]

print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")
print(f"✅ y_train shape: {y_train.shape}")

# Handle any remaining missing values (just in case)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# ============================================
# FEATURE SCALING
# ============================================

print("\n" + "=" * 60)
print("7. FEATURE SCALING")
print("=" * 60)

# Use RobustScaler (less sensitive to outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using RobustScaler")

# ============================================
# MODEL TRAINING
# ============================================

print("\n" + "=" * 60)
print("8. TRAINING MULTIPLE REGRESSION MODELS")
print("=" * 60)

# Log transform target (house prices are right-skewed)
y_train_log = np.log1p(y_train)

models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.0005),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                                                   max_depth=4, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train_log)
    
    # Cross-validation (negative MSE, so multiply by -1)
    cv_scores = cross_val_score(model, X_train_scaled, y_train_log, 
                                cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Training predictions
    train_pred_log = model.predict(X_train_scaled)
    train_pred = np.expm1(train_pred_log)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results.append({
        'Model': name,
        'CV RMSE': cv_rmse,
        'Train RMSE': train_rmse,
        'Train R²': train_r2
    })
    
    print(f"  CV RMSE: {cv_rmse:.4f}")
    print(f"  Train RMSE: ${train_rmse:,.0f}")
    print(f"  Train R²: {train_r2:.4f}")

results_df = pd.DataFrame(results).sort_values('CV RMSE')

print("\n" + "=" * 60)
print("MODEL COMPARISON (sorted by CV RMSE)")
print("=" * 60)
print(results_df.to_string(index=False))

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   CV RMSE: {results_df.iloc[0]['CV RMSE']:.4f}")

# ============================================
# ENSEMBLE PREDICTION (AVERAGING)
# ============================================

print("\n" + "=" * 60)
print("9. ENSEMBLE PREDICTION (MODEL AVERAGING)")
print("=" * 60)

print("Creating ensemble from all models...")

# Get predictions from all models (in log space)
predictions_log = []
for name, model in models.items():
    pred_log = model.predict(X_test_scaled)
    predictions_log.append(pred_log)

# Average predictions in log space
ensemble_pred_log = np.mean(predictions_log, axis=0)

# Convert back to original scale
ensemble_pred = np.expm1(ensemble_pred_log)

print(f"✅ Ensemble predictions created")
print(f"   Mean predicted price: ${ensemble_pred.mean():,.0f}")
print(f"   Median predicted price: ${np.median(ensemble_pred):,.0f}")

# ============================================
# CREATE SUBMISSIONS
# ============================================

print("\n" + "=" * 60)
print("10. CREATING SUBMISSION FILES")
print("=" * 60)

# Submission 1: Best single model
best_pred_log = best_model.predict(X_test_scaled)
best_pred = np.expm1(best_pred_log)

submission_best = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': best_pred
})
submission_best.to_csv('house_prices_submission_day22_best.csv', index=False)
print(f"✅ Created: house_prices_submission_day22_best.csv ({best_model_name})")

# Submission 2: Ensemble
submission_ensemble = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': ensemble_pred
})
submission_ensemble.to_csv('house_prices_submission_day22_ensemble.csv', index=False)
print(f"✅ Created: house_prices_submission_day22_ensemble.csv (Average of 5 models)")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "=" * 60)
print("11. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('House Prices Regression Analysis', fontsize=18, fontweight='bold')

# Plot 1: Model Comparison
ax1 = axes[0, 0]
models_sorted = results_df.sort_values('CV RMSE')
colors = ['lightcoral' if i == 0 else 'lightblue' for i in range(len(models_sorted))]
bars = ax1.barh(models_sorted['Model'], models_sorted['CV RMSE'], color=colors, edgecolor='black')
ax1.set_xlabel('Cross-Validation RMSE', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, models_sorted['CV RMSE'])):
    ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
            f'{val:.4f}', va='center', fontweight='bold')

# Plot 2: Actual vs Predicted (Best Model)
ax2 = axes[0, 1]
train_pred_log = best_model.predict(X_train_scaled)
train_pred = np.expm1(train_pred_log)
ax2.scatter(y_train, train_pred, alpha=0.5, s=20)
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Price ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax2.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = y_train - train_pred
ax3.scatter(train_pred, residuals, alpha=0.5, s=20)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
ax3.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Summary Stats
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
╔══════════════════════════════════════════╗
║     HOUSE PRICES REGRESSION PROJECT      ║
╠══════════════════════════════════════════╣
║                                          ║
║  DATASET:                                ║
║    Training: {len(train):,} houses                    ║
║    Test: {len(test):,} houses                       ║
║    Features: {X_train.shape[1]} (after engineering)         ║
║                                          ║
║  TARGET (SalePrice):                     ║
║    Mean: ${y_train.mean():,.0f}                    ║
║    Median: ${y_train.median():,.0f}                  ║
║    Range: ${y_train.min():,.0f} - ${y_train.max():,.0f}    ║
║                                          ║
║  BEST MODEL: {best_model_name:20s}   ║
║    CV RMSE: {results_df.iloc[0]['CV RMSE']:.4f}                      ║
║    Train R²: {results_df.iloc[0]['Train R²']:.4f}                     ║
║                                          ║
║  FEATURE ENGINEERING:                    ║
║    • Total square footage               ║
║    • Total bathrooms                    ║
║    • House age features                 ║
║    • Quality interaction features       ║
║    • One-hot encoded categoricals       ║
║                                          ║
║  ENSEMBLE:                               ║
║    Combined 5 models (averaging)        ║
║    Expected better generalization       ║
║                                          ║
║  NEXT STEPS:                             ║
║    • Submit to Kaggle                   ║
║    • Compare with leaderboard           ║
║    • Iterate & improve                  ║
║                                          ║
╚══════════════════════════════════════════╝

KEY LEARNINGS:
- Regression ≠ Classification
- Log transform for skewed targets
- Feature engineering matters
- Ensemble averaging reduces variance
- RobustScaler handles outliers better
"""

ax4.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig('plots/59_house_prices_regression.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: plots/59_house_prices_regression.png")

print("\n" + "=" * 60)
print("DAY 22: HOUSE PRICES REGRESSION COMPLETE!")
print("=" * 60)

print(f"""
PROJECT SUMMARY:
  • Dataset: 1,460 houses with 79 features
  • Target: Sale Price (regression problem)
  • Best Model: {best_model_name} (CV RMSE: {results_df.iloc[0]['CV RMSE']:.4f})
  • Features created: 13+ engineered features
  • Models trained: 5 (Ridge, Lasso, ElasticNet, RF, GB)
  • Ensemble: Model averaging for better predictions
  
SKILLS GAINED:
  • Regression techniques (vs classification)
  • Feature engineering for numerical data
  • Handling missing values strategically
  • Log transformation for skewed targets
  • Model ensembling (averaging)
  • RobustScaler for outlier handling

KAGGLE SUBMISSIONS READY:
  1. house_prices_submission_day22_best.csv (Best single model)
  2. house_prices_submission_day22_ensemble.csv (Ensemble)

NEXT: Submit both to Kaggle and compare scores!
""")
