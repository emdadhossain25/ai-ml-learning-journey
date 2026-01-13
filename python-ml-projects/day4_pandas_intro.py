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