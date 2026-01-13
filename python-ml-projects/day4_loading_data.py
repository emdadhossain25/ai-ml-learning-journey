"""
Day 4: Loading and Exploring Real Data
Working with the Titanic dataset
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("LOADING CSV DATA")
print("=" * 60)

df =pd.read_csv('data/titanic.csv')

print("\n Data Loaded Successfully")
print(f"Shape: {df.shape} ")
print(f"We Have {df.shape[0]} passengers and {df.shape[1]} features ")

print("\n" + "=" * 60)
print("FIRST LOOK AT THE DATA")
print("=" * 60)

print("\nfirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\n" + "=" * 60)
print("UNDERSTANDING THE DATASET")
print("=" * 60)


print("\nDataset Info:")
df.info()

print("\nStatistical summary:")
df.describe()

print("\n" + "=" * 60)
print("CHECKING FOR MISSING DATA - CRITICAL!")
print("=" * 60)

print("\nMissing Values per column:")
missing = df.isnull().sum()
print(missing[missing>0])


print("\nPercentage of missing data:")
missing_percent= (df.isnull().sum() / len(df))*100
print(missing_percent[missing_percent>0])

print("\n" + "=" * 60)
print("EXPLORING SPECIFIC COLUMNS")
print("=" * 60)


#survival rate
print(f"Survival Rate: {df['Survived'].mean():.2%}")

#Age Distributio 
print("\nAge Statistics")
print(df['Age'].describe())


# Class distribution
print(f"\nPassengers by class:")
print(df['Pclass'].value_counts())

# Gender distribution
print(f"\nPassengers by gender:")
print(df['Sex'].value_counts())

print("\n" + "=" * 60)
print("ASKING QUESTIONS WITH DATA")
print("=" * 60)

# Question 1: Did women survive more than men?
print("Survival rate by gender:")
print(df.groupby('Sex')['Survived'].mean())

# Question 2: Did class affect survival?
print("\nSurvival rate by class:")
print(df.groupby('Pclass')['Survived'].mean())

# Question 3: Average fare by class
print("\nAverage fare by class:")
print(df.groupby('Pclass')['Fare'].mean())

# Question 4: How many children survived?
children = df[df['Age'] < 18]
print(f"\nChild survival rate: {children['Survived'].mean():.2%}")
print(f"Adult survival rate: {df[df['Age'] >= 18]['Survived'].mean():.2%}")

print("\n" + "=" * 60)
print("CREATING NEW FEATURES (Feature Engineering)")
print("=" * 60)

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Senior'])

print("Survival by age group:")
print(df.groupby('AgeGroup')['Survived'].mean())

# Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("\nSurvival by family size:")
print(df.groupby('FamilySize')['Survived'].mean())

# Fare per person
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

print("\n✅ Real dataset exploration complete!")











