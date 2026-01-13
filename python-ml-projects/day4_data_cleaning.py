"""
Day 4: Data Cleaning Techniques
Handling missing data, outliers, and inconsistencies
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("DATA CLEANING - THE ESSENTIAL ML SKILL")
print("=" * 60)

#Load Data
df = pd.read_csv('data/titanic.csv')
print(f"Origin Shape:{df.shape}")


print("\n" + "=" * 60)
print("HANDLING MISSING DATA")
print("=" * 60)


#Check Missing Values
print("Missing values:")
print(df.isnull().sum())


#Stratergy1: Drop Rows with any missing values (aggressive)
df_dropna =df.dropna()
print(f"\nAfter Dropping All Missing:{df_dropna.shape}")
print("We Lost Too Much Data, Not Recommended")

#Stratergy2: Drop Rows with missing values in specific columns
df_dropped=df.dropna(subset=['Age','Embarked'])
print(f"\nAfter Dropping Rows with Missing Age/Embarked:{df_dropped.shape}")

# Strategy 3: Fill missing values (BEST for most cases)
df_filled = df.copy()

#Fill Age with median , Robust to Outliers
median_age = df_filled['Age'].median()
df_filled['Age'].fillna(median_age,inplace=True)
print(f"\nFilled Age with median: {median_age}")

# Fill Embarked with mode (most common)
mode_embarked = df_filled['Embarked'].mode()[0]
df_filled['Embarked'].fillna(mode_embarked, inplace=True)
print(f"Filled Embarked with mode: {mode_embarked}")

# Fill Cabin with 'Unknown'
df_filled['Cabin'].fillna('Unknown', inplace=True)

print("\nMissing values after filling:")
print(df_filled.isnull().sum())

print("\n" + "=" * 60)
print("HANDLING DUPLICATES")
print("=" * 60)

print(f"Duplicate rows: {df_filled.duplicated().sum()}")


# Remove duplicates
df_clean = df_filled.drop_duplicates()
print(f"Shape after removing duplicates: {df_clean.shape}")



print("\n" + "=" * 60)
print("HANDLING OUTLIERS")
print("=" * 60)

# Check for outliers in Fare
print("Fare statistics:")
print(df_clean['Fare'].describe())

# Find outliers using IQR method
Q1 = df_clean['Fare'].quantile(0.25)
Q3 = df_clean['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR: {IQR:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

outliers = df_clean[(df_clean['Fare'] < lower_bound) | (df_clean['Fare'] > upper_bound)]
print(f"Number of outliers in Fare: {len(outliers)}")

# Option 1: Remove outliers
df_no_outliers = df_clean[(df_clean['Fare'] >= lower_bound) & (df_clean['Fare'] <= upper_bound)]
print(f"Shape after removing outliers: {df_no_outliers.shape}")

# Option 2: Cap outliers (better for ML)
df_capped = df_clean.copy()
df_capped['Fare'] = df_capped['Fare'].clip(lower=lower_bound, upper=upper_bound)
print("✅ Outliers capped to bounds")

print("\n" + "=" * 60)
print("DATA TYPE CONVERSIONS")
print("=" * 60)

# Convert Survived to categorical
df_clean['Survived'] = df_clean['Survived'].astype('category')
df_clean['Pclass'] = df_clean['Pclass'].astype('category')

print("Updated data types:")
print(df_clean.dtypes)

print("\n" + "=" * 60)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

# One-hot encoding for Sex
df_encoded = pd.get_dummies(df_clean, columns=['Sex', 'Embarked'], prefix=['Sex', 'Port'])

print("Columns after encoding:")
print(df_encoded.columns.tolist())
print(f"\nNew shape: {df_encoded.shape}")

# Label encoding for ordinal data (Pclass is ordered)
df_encoded['Pclass'] = df_encoded['Pclass'].cat.codes

print("\n" + "=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)

# Select relevant columns for ML
features_for_ml = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Sex_female', 'Sex_male', 
                   'Port_C', 'Port_Q', 'Port_S']

df_final = df_encoded[features_for_ml + ['Survived']].copy()

print("Final dataset for ML:")
print(df_final.head())
print(f"\nFinal shape: {df_final.shape}")
print(f"Missing values: {df_final.isnull().sum().sum()}")

# Save cleaned data
df_final.to_csv('data/titanic_cleaned.csv', index=False)
print("\n✅ Clean data saved to 'data/titanic_cleaned.csv'")

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)
print(f"Original rows: {len(df)}")
print(f"Final rows: {len(df_final)}")
print(f"Original columns: {df.shape[1]}")
print(f"Final columns: {df_final.shape[1]}")
print("\n✅ Data is now ready for Machine Learning!")