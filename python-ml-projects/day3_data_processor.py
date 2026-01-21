"""
Day 3: Mini Project - Data Processor
Your first step toward building ML pipelines
"""


import numpy as np

class DataProcessor:
    def __init__(self, data):
        """Initialize with dataset"""
        self.data = np.array(data)
        self.normalized_data = None

    def summary(self):
        """Print summary statistics of the dataset"""
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Shape:{self.data.shape}")
        print(f"Samples: {self.data.shape[0]}")
        print(f"Features: {self.data.shape[1]}")
        print(f"First 5 rows:\n{self.data[:5]}")
        print(f"Mean per feature: {np.mean(self.data, axis=0)}")
        print(f"Std per feature: {np.std(self.data, axis = 0)}")
        print(f"Min per feature: {np.min(self.data, axis = 0)}")
        print(f"Max per feature: {np.max(self.data, axis = 0)}")

    def normalize_minmax(self):
        """Normalize dataset using Min-Max Scaling"""
        min_vals = np.min(self.data, axis=0)
        max_vals = np.max(self.data, axis=0)
        self.normalized_data = (self.data - min_vals) / (max_vals - min_vals)
        print("\nDataset normalized using Min-Max Scaling.")
        return self.normalized_data

    def normalize_zscore(self):
        """Normalize dataset using Z-score Normalization"""
        mean_vals = np.mean(self.data, axis=0)
        std_vals = np.std(self.data, axis=0)
        self.normalized_data = (self.data - mean_vals) / std_vals
        print("\nDataset normalized using Z-score Normalization.")
        return self.normalized_data

    def remove_outliers(self, threshold=3):
        """Remove outliers beyond a certain z-score threshold"""
        mean_vals = np.mean(self.data, axis=0)
        std_vals = np.std(self.data, axis=0)
        z_scores = np.abs((self.data - mean_vals) / std_vals)
        mask = np.all(z_scores < threshold, axis=1)
        self.data = self.data[mask]
        print(f"\n✅ Removed outliers. New shape: {self.data.shape}")
        return self.data

    def train_test_split(self, test_ratio=0.2):
        """Split dataset into training and testing sets"""
        np.random.shuffle(self.data)
        #Split
        split_idx = int(len(self.data) * (1 - test_ratio))
        train = self.data[:split_idx]
        test = self.data[split_idx:]
        print(f"\n✅ Data split:")
        print(f"  Training: {train.shape}")
        print(f"  Test: {test.shape}")
        
        return train, test

    def add_polynomial_features(self, degree=2):
        """ Add polynomial features (feature engineering)"""
        squared = self.data ** 2
        self.data = np.hstack([self.data, squared])
        print(f"\n✅ Added polynomial features. New shape: {self.data.shape}")
        return self.data


# ============================================
# DEMONSTRATION
# ============================================

print("=" * 60)
print("DATA PROCESSOR DEMO")
print("=" * 60)

# Create sample dataset
# Features: [age, income, credit_score, years_employed]
raw_data = [
    [25, 50000, 720, 2],
    [35, 75000, 780, 8],
    [45, 60000, 650, 15],
    [28, 55000, 700, 3],
    [52, 95000, 800, 20],
    [31, 68000, 740, 6],
    [40, 72000, 710, 12],
    [29, 58000, 690, 4],
    [38, 82000, 770, 10],
    [26, 52000, 680, 1],
]

# Initialize processor
processor = DataProcessor(raw_data)

# Step 1: See the data
processor.summary()

# Step 2: Normalize
normalized = processor.normalize_minmax()
print("\nNormalized data (first 3 rows):")
print(normalized[:3])

# Step 3: Train-test split
train, test = processor.train_test_split(test_ratio=0.2)

print("\nTraining set:")
print(train)
print("\nTest set:")
print(test)

print("\n" + "=" * 60)
print("REAL-WORLD EXAMPLE: House Prices")
print("=" * 60)

# Features: [square_feet, bedrooms, bathrooms, age_years]
house_data = np.array([
    [1500, 3, 2, 10],
    [2200, 4, 3, 5],
    [1800, 3, 2, 15],
    [2500, 4, 3, 8],
    [1200, 2, 1, 20],
    [3000, 5, 4, 2],
    [1600, 3, 2, 12],
    [2000, 3, 2.5, 7],
])

house_processor = DataProcessor(house_data)
house_processor.summary()

# Prepare for ML
normalized_houses = house_processor.normalize_zscore()
train_houses, test_houses = house_processor.train_test_split(0.25)

print("\n✅ Data ready for machine learning!")
print(f"Training on {len(train_houses)} houses")
print(f"Testing on {len(test_houses)} houses")