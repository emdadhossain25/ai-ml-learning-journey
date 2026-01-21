"""
Day 4: Pandas Fundamentals
The #1 tool for data manipulation in ML
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("PANDAS SERIES - 1D Labeled Data")
print("=" * 60)

#Series = 1D array with labels (like a column in a spreadsheet)
ages = pd.Series([25, 30, 35, 40], index=['Alice', 'Bob', 'Carol', 'Dave'])
print("Ages Series:")
print(ages)
print(f"\nBob's age: {ages['Bob']}")
print(f"\nMean age: {ages.mean()}")

print("=" * 60)
print("PANDAS DATAFRAME - The Star of the Show")
print("=" * 60)

# DataFrame = 2D table with labeled rows and columns
data = {
    'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'income': [50000, 75000, 60000, 95000, 68000],
    'city': ['NYC', 'LA', 'Chicago', 'Boston', 'NYC']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)


print("\n" + "=" * 60)
print("DATAFRAME BASIC INFO")
print("=" * 60)

print(f"\nShape: {df.shape}") #(rows, columns)
print(f"Columns: {df.columns.tolist()}")
print("\nData Types: \n{df.dtypes}")

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))


print("\nQuick Statistics:")
print(df.describe())

print("Dataframe Info:")
print(df.info())

print("\n" + "=" * 60)
print("SELECTING DATA - Different Ways")
print("=" * 60)

# Select a column (returns Series)
print("All ages:")
print(df['age'])

# Select multiple columns (returns DataFrame)
print("\nNames and Incomes:")
print(df[['name', 'income']])


#Select rows by position
print("\nFirst 2 rows:")
print(df[0:2])

# Select by label (.loc)
print("\nRow 0:")
print(df.loc[0])


# Select by position (.iloc)
print("First 2 rows First 2 columns:")
print(df.iloc[0:2, 0:2])


#Select specific cells
print("\nAlice's income:")
print(df.loc[0, 'income'])



print("\n" + "=" * 60)
print("Filtering Data - Boolean Indexing")
print("=" * 60)

# Filter rows where age > 30
print("\nPeople older than 30:")
print(df[df['age'] > 30])


# People in NYC
print("\nPeople in NYC:")
print(df[df['city'] == 'NYC'])

# Multiple Conditions
print("\nPeople older than 28 in NYC:")
print(df[(df['age'] > 28) & (df['city'] == 'NYC')])


#High Earners
print("\nPeople earning more than 60000:")
print(df[df['income'] > 60000])    

print("\n" + "=" * 60)
print("ADDING AND MODIFYING COLUMNS")
print("=" * 60)

# Add a new column
df['income_k'] = df['income'] / 1000
print("Added Income in Thousands:")
print(df)

# Calculate New Column From Existing Columns
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Mid' if x < 40 else 'Senior')
print("\nWith Age Group:")
print(df)    


# Modify existing column
df['income'] = df['income'] * 1.1  # 10% increase
print("\n After 10% raise:")
print(df[['name', 'income']])


print("\n" + "=" * 60)
print("GROUPING AND AGGREGATION")
print("=" * 60)

#Group by city
print("\nAverage income by city:")
print(df.groupby('city')['income'].mean())

print("\nCount by city:")
print(df.groupby('city').size())

print("\n Multiple aggregations:")
print(df.groupby('city').agg({'income': ['mean', 'max'], 'age': 'min'}))



print("\n" + "=" * 60)
print("SORTING DATA")
print("=" * 60)

# Sort by age
print("\nSorted by age(asceding):")
print(df.sort_values(by='age'))


# Sort by income descending
print("\nSorted by income(descending):")
print(df.sort_values(by='income', ascending=False))

#Sort by multiple columns
print("\nSorted by city and then age:")
print(df.sort_values(by=['city', 'age']))

print("\n✅ Pandas basics complete!")








