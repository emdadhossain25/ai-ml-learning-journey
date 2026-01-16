"""
Day 7: Model Comparison
Comparing different classification algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)

print("=" * 60)
print("COMPARING CLASSIFICATION MODELS")
print("=" * 60)

# Load Titanic data
df = pd.read_csv('data/titanic.csv')

# Prepare data (simplified for comparison)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Select features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train)} passengers")
print(f"Test set: {len(X_test)} passengers")

# ============================================
# DEFINE MODELS TO COMPARE
# ============================================

print("\n" + "=" * 60)
print("TRAINING MULTIPLE MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Store results
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    print(f"  ✅ {name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

# ============================================
# RESULTS COMPARISON
# ============================================

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + results_df.to_string(index=False))

print(f"\n🏆 BEST MODEL: {results_df.iloc[0]['Model']}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"   ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("CREATING COMPARISON VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Classification Model Comparison', fontsize=18, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
bars = ax1.barh(results_df['Model'], results_df['Accuracy'], color='skyblue', edgecolor='black')
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, results_df['Accuracy'])):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Plot 2: All Metrics Comparison
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15

for i, metric in enumerate(metrics):
    offset = width * (i - 2)
    ax2.bar(x + offset, results_df[metric], width, label=metric, alpha=0.8)

ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Cross-Validation Scores
ax3 = axes[1, 0]
ax3.bar(results_df['Model'], results_df['CV Mean'], 
       yerr=results_df['CV Std'], capsize=5, color='lightgreen', edgecolor='black')
ax3.set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Precision vs Recall
ax4 = axes[1, 1]
ax4.scatter(results_df['Recall'], results_df['Precision'], s=200, alpha=0.6)
for i, model in enumerate(results_df['Model']):
    ax4.annotate(model, (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]),
                fontsize=9, ha='center')
ax4.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax4.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.grid(alpha=0.3)
ax4.plot([0, 1], [0, 1], 'r--', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/31_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/31_model_comparison.png")

# ============================================
# UNDERSTANDING METRICS
# ============================================

print("\n" + "=" * 60)
print("UNDERSTANDING CLASSIFICATION METRICS")
print("=" * 60)

print("""
ACCURACY: Overall correctness
  = (Correct Predictions) / (Total Predictions)
  Good when classes are balanced

PRECISION: Of predicted positives, how many are correct?
  = True Positives / (True Positives + False Positives)
  Important when false positives are costly (spam detection)

RECALL (Sensitivity): Of actual positives, how many did we catch?
  = True Positives / (True Positives + False Negatives)
  Important when false negatives are costly (disease detection)

F1-SCORE: Harmonic mean of Precision and Recall
  = 2 * (Precision * Recall) / (Precision + Recall)
  Good overall metric

ROC-AUC: Area Under ROC Curve
  Measures model's ability to discriminate between classes
  1.0 = Perfect, 0.5 = Random
""")

print("\n" + "=" * 60)
print("🎉 MODEL COMPARISON COMPLETE!")
print("=" * 60)
print(f"\nBest performing model: {results_df.iloc[0]['Model']}")
print("You now understand multiple classification algorithms!")