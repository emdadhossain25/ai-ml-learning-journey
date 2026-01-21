
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic credit card fraud dataset
print("Generating synthetic fraud detection dataset...")

np.random.seed(42)

# Create highly imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.98, 0.02],  # 2% fraud
    flip_y=0.01,
    random_state=42
)

# Create feature names
feature_names = [f'V{i}' for i in range(1, 21)]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['Amount'] = np.random.exponential(100, 10000)
df['Class'] = y

# Save
df.to_csv('data/credit_fraud.csv', index=False)
print(f"✅ Dataset created: {len(df)} transactions")
print(f"   Legitimate: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"   Fraud: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)")
