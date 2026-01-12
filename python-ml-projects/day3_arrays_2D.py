"""
Day 3: 2D Arrays - Real ML Dataset Structure
This is how ML libraries see data
"""
import numpy as np
print("=" * 60)
print("2D ARRAYS - THE ML DATASET FORMAT")
print("=" * 60)


# Think of this as a dataset with 4 samples, 3 features each
# Rows = samples/examples, Columns = features

dataset = np.array([
    [25, 50000, 720],  # Person 1: age, income, credit_score
    [35, 75000, 780],  # Person 2
    [45, 60000, 650],  # Person 3
    [28, 55000, 700]   # Person 4
])

print("Dataset (4 people with 3 features each):")
print(dataset)
print(f"\nShape of dataset: {dataset.shape} (4 samples, 3 features each)")
print(f"This means {dataset.shape[0]} samples and {dataset.shape[1]} features per sample.")

print("\n" + "=" * 60)
print("ACCESSING DATA - INDEXING")
print("=" * 60)

print(f"\nFirst sample (Person 1): {dataset[0]}")
print(f"\nSecond sample (Person 2): {dataset[1]}")

print(f"\n All ages (first column): {dataset[:, 0]} ")
print(f"\n All incomes (second column): {dataset[:, 1]} ")
print(f"\n All credit scores (third column): {dataset[:, 2]} ")


print(f"\ first two people : \n {dataset[:2]}")
print(f"\ last two people : \n {dataset[:-2]}")


#Specific Element
print(f"\n Person 3's income: {dataset[2,1]}")  # 60000

print("\n" + "=" * 60)
print("FEATURE STATISTICS - CRITICAL FOR ML")
print("=" * 60)

#statistics along axis
print(f"\n Mean of each feature (column): {np.mean(dataset, axis = 0)}")  # Mean age, income, credit_score
print(f"\n -Average age: {np.mean(dataset[:,0]):.1f}")
print(f"\n -Average income: {np.mean(dataset[:,1]):.0f}")
print(f"\n -Average credit score: {np.mean(dataset[:,2]):.0f}")

print(f"\n Standard Deviation of each feature (column): {np.std(dataset, axis = 0)}")
print(f"\n  Minimum of each feature (column): {np.min(dataset, axis = 0)}")
print(f"\n  Maximum of each feature (column): {np.max(dataset, axis = 0)}")


print("\n" + "=" * 60)
print("NORMALIZATION - KEY ML PREPROCESSING")
print("=" * 60)

# Min-Max Normalization to scale features between 0 and 1
def normalize(data):
    """Normalize each feature to [0, 1] range using Min-Max Scaling."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

normalized_data = normalize(dataset)
print(f"\nNormalized Dataset:\n{normalized_data}")
print(f"\n Now all features are on same scale")




print("\n" + "=" * 60)
print("BOOLEAN INDEXING - FILTERING DATA")
print("=" * 60)

# Find people with income > 60000
high_income = dataset[:,1] > 60000
print(f"\n high income mask: {high_income}")
print(f"people with high income:\n {dataset[high_income]}")


#multiple conditions 
good_credit = dataset[:,2] >700
young_and_good_credit = (dataset[:,0] <40) & (good_credit)
print(f"\n young and good credit : {dataset[young_and_good_credit]}")

print("\n" + "=" * 60)
print("ARRAY MANIPULATION")
print("=" * 60)

flat = np.arange(12)
print(f"Flat array: {flat}")

reshaped = flat.reshape((3,4))
print(f"\n Reshaped to 3x4:\n {reshaped}")

reshaped2 = flat.reshape((4,3))
print(f"\n Reshaped to 4x3:\n {reshaped2}")

# Transpose (swap rows and columns)
print(f"\n original shape: {dataset.shape}")
transposed = dataset.T
print(f"\n Transposed dataset shape: {transposed.shape}")
print(f"\n Transposed dataset:\n {transposed}")

print("\n" + "=" * 60)
print("CONCATENATION - COMBINING DATA")
print("=" * 60)

new_people = np.array([
    [22, 45000, 680],
    [50, 90000, 800]
])

combined = np.vstack((dataset, new_people)) #Stack vertically (add rows)
print(f"\n Added 2 more people:\n {combined}")
print(f"\n New Shape:\n {combined.shape}")


#Add a new feature (column) - e.g., years of education
ages_in_months = dataset[:,0:1] * 12
print(f"\n Ages in months:\n {ages_in_months}")



dataset_extended = np.hstack((dataset, ages_in_months))
print(f"\n Dataset with new feature (ages in months):\n {dataset_extended}")

print("\n✅ 2D arrays complete! This is how ML sees data.")

