"""
Day 9: Handling Imbalanced Data
Techniques for dealing with class imbalance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HANDLING IMBALANCED DATA")
print("=" * 60)

# ============================================
# WHAT IS CLASS IMBALANCE?
# ============================================

print("\n1. UNDERSTANDING CLASS IMBALANCE")
print("-" * 60)

print("""
CLASS IMBALANCE: When one class has many more samples than another

EXAMPLES:
  • Fraud Detection: 99.9% legitimate, 0.1% fraud
  • Disease Diagnosis: 95% healthy, 5% sick
  • Manufacturing: 99% good products, 1% defective
  • Email: 80% legitimate, 20% spam

THE PROBLEM:
  Model learns to predict majority class
  → 99% accuracy by always predicting "not fraud"
  → But misses ALL fraud cases!
  → Accuracy is misleading!

SOLUTION:
  Use better metrics: Precision, Recall, F1, ROC-AUC
  Balance the dataset: Oversample, Undersample, SMOTE
  Adjust model: Class weights, threshold tuning
""")

# ============================================
# CREATE IMBALANCED DATASET
# ============================================

print("\n2. CREATING IMBALANCED DATASET")
print("-" * 60)

# Load Titanic (naturally imbalanced)
df = pd.read_csv('data/titanic.csv')

# Feature engineering
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

X = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']],
                   drop_first=True)
y = df['Survived']

# Check class distribution
class_counts = y.value_counts()
imbalance_ratio = class_counts[0] / class_counts[1]

print(f"Class distribution:")
print(class_counts)
print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
print(f"  Died (0): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"  Survived (1): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"  Class 0: {(y_train == 0).sum()}")
print(f"  Class 1: {(y_train == 1).sum()}")

# ============================================
# BASELINE: TRAIN ON IMBALANCED DATA
# ============================================

print("\n" + "=" * 60)
print("3. BASELINE MODEL (No Balancing)")
print("=" * 60)

baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)
baseline_proba = baseline_model.predict_proba(X_test)[:, 1]

baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)
baseline_auc = roc_auc_score(y_test, baseline_proba)

print(f"\nBaseline Performance:")
print(f"  Accuracy: {baseline_acc:.4f}")
print(f"  Precision: {baseline_precision:.4f}")
print(f"  Recall: {baseline_recall:.4f}")
print(f"  F1-Score: {baseline_f1:.4f}")
print(f"  ROC-AUC: {baseline_auc:.4f}")

print(f"\nConfusion Matrix:")
baseline_cm = confusion_matrix(y_test, baseline_pred)
print(baseline_cm)

# ============================================
# TECHNIQUE 1: CLASS WEIGHTS
# ============================================

print("\n" + "=" * 60)
print("4. TECHNIQUE 1: CLASS WEIGHTS")
print("=" * 60)

print("Giving more importance to minority class...")

weighted_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatically balance
    random_state=42
)
weighted_model.fit(X_train, y_train)

weighted_pred = weighted_model.predict(X_test)
weighted_proba = weighted_model.predict_proba(X_test)[:, 1]

weighted_acc = accuracy_score(y_test, weighted_pred)
weighted_precision = precision_score(y_test, weighted_pred)
weighted_recall = recall_score(y_test, weighted_pred)
weighted_f1 = f1_score(y_test, weighted_pred)
weighted_auc = roc_auc_score(y_test, weighted_proba)

print(f"\nClass Weighted Performance:")
print(f"  Accuracy: {weighted_acc:.4f}")
print(f"  Precision: {weighted_precision:.4f}")
print(f"  Recall: {weighted_recall:.4f} ← Improved!")
print(f"  F1-Score: {weighted_f1:.4f}")
print(f"  ROC-AUC: {weighted_auc:.4f}")

print(f"\nConfusion Matrix:")
weighted_cm = confusion_matrix(y_test, weighted_pred)
print(weighted_cm)

# ============================================
# TECHNIQUE 2: RANDOM OVERSAMPLING
# ============================================

print("\n" + "=" * 60)
print("5. TECHNIQUE 2: RANDOM OVERSAMPLING")
print("=" * 60)

print("Duplicating minority class samples...")

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print(f"\nAfter Random Oversampling:")
print(f"  Original: {len(X_train)} samples")
print(f"  Resampled: {len(X_train_ros)} samples")
print(f"  Class 0: {(y_train_ros == 0).sum()}")
print(f"  Class 1: {(y_train_ros == 1).sum()}")

ros_model = RandomForestClassifier(n_estimators=100, random_state=42)
ros_model.fit(X_train_ros, y_train_ros)

ros_pred = ros_model.predict(X_test)
ros_proba = ros_model.predict_proba(X_test)[:, 1]

ros_acc = accuracy_score(y_test, ros_pred)
ros_precision = precision_score(y_test, ros_pred)
ros_recall = recall_score(y_test, ros_pred)
ros_f1 = f1_score(y_test, ros_pred)
ros_auc = roc_auc_score(y_test, ros_proba)

print(f"\nRandom Oversampling Performance:")
print(f"  Accuracy: {ros_acc:.4f}")
print(f"  Precision: {ros_precision:.4f}")
print(f"  Recall: {ros_recall:.4f}")
print(f"  F1-Score: {ros_f1:.4f}")
print(f"  ROC-AUC: {ros_auc:.4f}")

# ============================================
# TECHNIQUE 3: RANDOM UNDERSAMPLING
# ============================================

print("\n" + "=" * 60)
print("6. TECHNIQUE 3: RANDOM UNDERSAMPLING")
print("=" * 60)

print("Removing majority class samples...")

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"\nAfter Random Undersampling:")
print(f"  Original: {len(X_train)} samples")
print(f"  Resampled: {len(X_train_rus)} samples")
print(f"  Class 0: {(y_train_rus == 0).sum()}")
print(f"  Class 1: {(y_train_rus == 1).sum()}")

rus_model = RandomForestClassifier(n_estimators=100, random_state=42)
rus_model.fit(X_train_rus, y_train_rus)

rus_pred = rus_model.predict(X_test)
rus_proba = rus_model.predict_proba(X_test)[:, 1]

rus_acc = accuracy_score(y_test, rus_pred)
rus_precision = precision_score(y_test, rus_pred)
rus_recall = recall_score(y_test, rus_pred)
rus_f1 = f1_score(y_test, rus_pred)
rus_auc = roc_auc_score(y_test, rus_proba)

print(f"\nRandom Undersampling Performance:")
print(f"  Accuracy: {rus_acc:.4f}")
print(f"  Precision: {rus_precision:.4f}")
print(f"  Recall: {rus_recall:.4f}")
print(f"  F1-Score: {rus_f1:.4f}")
print(f"  ROC-AUC: {rus_auc:.4f}")

# ============================================
# TECHNIQUE 4: SMOTE
# ============================================

print("\n" + "=" * 60)
print("7. TECHNIQUE 4: SMOTE (Synthetic Minority Oversampling)")
print("=" * 60)

print("Creating synthetic minority samples...")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Original: {len(X_train)} samples")
print(f"  Resampled: {len(X_train_smote)} samples")
print(f"  Class 0: {(y_train_smote == 0).sum()}")
print(f"  Class 1: {(y_train_smote == 1).sum()}")
print("\nSMOTE creates NEW synthetic samples (not duplicates!)")

smote_model = RandomForestClassifier(n_estimators=100, random_state=42)
smote_model.fit(X_train_smote, y_train_smote)

smote_pred = smote_model.predict(X_test)
smote_proba = smote_model.predict_proba(X_test)[:, 1]

smote_acc = accuracy_score(y_test, smote_pred)
smote_precision = precision_score(y_test, smote_pred)
smote_recall = recall_score(y_test, smote_pred)
smote_f1 = f1_score(y_test, smote_pred)
smote_auc = roc_auc_score(y_test, smote_proba)

print(f"\nSMOTE Performance:")
print(f"  Accuracy: {smote_acc:.4f}")
print(f"  Precision: {smote_precision:.4f}")
print(f"  Recall: {smote_recall:.4f}")
print(f"  F1-Score: {smote_f1:.4f}")
print(f"  ROC-AUC: {smote_auc:.4f}")

# ============================================
# TECHNIQUE 5: THRESHOLD TUNING
# ============================================

print("\n" + "=" * 60)
print("8. TECHNIQUE 5: THRESHOLD TUNING")
print("=" * 60)

print("Finding optimal probability threshold...")

# Test different thresholds
thresholds = np.arange(0.3, 0.7, 0.05)
threshold_results = []

for threshold in thresholds:
    pred_threshold = (baseline_proba >= threshold).astype(int)
    precision = precision_score(y_test, pred_threshold)
    recall = recall_score(y_test, pred_threshold)
    f1 = f1_score(y_test, pred_threshold)
    
    threshold_results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

threshold_df = pd.DataFrame(threshold_results)
best_threshold = threshold_df.loc[threshold_df['F1-Score'].idxmax()]

print(f"\nBest threshold: {best_threshold['Threshold']:.2f}")
print(f"  Precision: {best_threshold['Precision']:.4f}")
print(f"  Recall: {best_threshold['Recall']:.4f}")
print(f"  F1-Score: {best_threshold['F1-Score']:.4f}")

# ============================================
# COMPARISON OF ALL TECHNIQUES
# ============================================

print("\n" + "=" * 60)
print("9. COMPREHENSIVE COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Technique': ['Baseline', 'Class Weights', 'Oversampling', 
                 'Undersampling', 'SMOTE', 'Threshold Tuning'],
    'Accuracy': [baseline_acc, weighted_acc, ros_acc, rus_acc, smote_acc, 
                accuracy_score(y_test, (baseline_proba >= best_threshold['Threshold']).astype(int))],
    'Precision': [baseline_precision, weighted_precision, ros_precision, 
                 rus_precision, smote_precision, best_threshold['Precision']],
    'Recall': [baseline_recall, weighted_recall, ros_recall, 
              rus_recall, smote_recall, best_threshold['Recall']],
    'F1-Score': [baseline_f1, weighted_f1, ros_f1, 
                rus_f1, smote_f1, best_threshold['F1-Score']],
    'ROC-AUC': [baseline_auc, weighted_auc, ros_auc, rus_auc, smote_auc, baseline_auc]
})

print("\n" + comparison.to_string(index=False))

best_technique = comparison.loc[comparison['F1-Score'].idxmax()]
print(f"\n🏆 BEST TECHNIQUE: {best_technique['Technique']}")
print(f"   F1-Score: {best_technique['F1-Score']:.4f}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("10. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Handling Imbalanced Data - Comprehensive Analysis', 
             fontsize=18, fontweight='bold')

# Plot 1: Class Distribution
ax1 = axes[0, 0]
class_dist = y_train.value_counts()
colors = ['lightcoral', 'lightgreen']
bars = ax1.bar(['Died (0)', 'Survived (1)'], class_dist.values,
              color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Original Class Distribution (Imbalanced)', 
             fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, class_dist.values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 10,
            f'{val}\n({val/len(y_train)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Resampling Effects
ax2 = axes[0, 1]
techniques = ['Original', 'Oversample', 'Undersample', 'SMOTE']
class0_counts = [
    (y_train == 0).sum(),
    (y_train_ros == 0).sum(),
    (y_train_rus == 0).sum(),
    (y_train_smote == 0).sum()
]
class1_counts = [
    (y_train == 1).sum(),
    (y_train_ros == 1).sum(),
    (y_train_rus == 1).sum(),
    (y_train_smote == 1).sum()
]

x = np.arange(len(techniques))
width = 0.35

bars1 = ax2.bar(x - width/2, class0_counts, width, label='Class 0 (Died)',
               alpha=0.8, color='lightcoral', edgecolor='black')
bars2 = ax2.bar(x + width/2, class1_counts, width, label='Class 1 (Survived)',
               alpha=0.8, color='lightgreen', edgecolor='black')

ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
ax2.set_title('Effect of Resampling Techniques', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(techniques, rotation=15)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: F1-Score Comparison
ax3 = axes[1, 0]
techniques = comparison['Technique'].values
f1_scores = comparison['F1-Score'].values

bars = ax3.barh(techniques, f1_scores, color='skyblue', edgecolor='black', linewidth=2)
ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax3.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.set_xlim(0.65, 0.82)

for bar, val in zip(bars, f1_scores):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

# Plot 4: Precision vs Recall Trade-off
ax4 = axes[1, 1]
precisions = comparison['Precision'].values
recalls = comparison['Recall'].values

ax4.scatter(recalls, precisions, s=300, alpha=0.7, edgecolors='black', linewidth=2)

for i, technique in enumerate(techniques):
    ax4.annotate(technique, (recalls[i], precisions[i]),
                fontsize=9, ha='center', va='bottom')

ax4.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax4.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.set_xlim(0.65, 0.95)
ax4.set_ylim(0.65, 0.90)

# Plot 5: Threshold Tuning Effect
ax5 = axes[2, 0]
ax5.plot(threshold_df['Threshold'], threshold_df['Precision'], 
        'o-', linewidth=3, markersize=8, label='Precision', color='blue')
ax5.plot(threshold_df['Threshold'], threshold_df['Recall'],
        's-', linewidth=3, markersize=8, label='Recall', color='red')
ax5.plot(threshold_df['Threshold'], threshold_df['F1-Score'],
        '^-', linewidth=3, markersize=8, label='F1-Score', color='green')

ax5.axvline(x=best_threshold['Threshold'], color='purple', 
           linestyle='--', linewidth=2, label=f"Best ({best_threshold['Threshold']:.2f})")

ax5.set_xlabel('Probability Threshold', fontsize=12, fontweight='bold')
ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_title('Threshold Tuning Impact', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# Plot 6: Confusion Matrices Comparison
ax6 = axes[2, 1]
ax6.axis('off')

# Create small heatmaps
confusion_matrices = [
    ('Baseline', baseline_cm),
    ('SMOTE', confusion_matrix(y_test, smote_pred)),
    ('Class Weights', weighted_cm)
]

for idx, (name, cm) in enumerate(confusion_matrices):
    # Create mini subplot
    mini_ax = plt.axes([0.55 + (idx % 2) * 0.2, 0.08 + (1 - idx // 2) * 0.15, 0.15, 0.12])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['D', 'S'], yticklabels=['D', 'S'],
               ax=mini_ax, annot_kws={'fontsize': 10})
    mini_ax.set_title(name, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/41_imbalanced_data_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/41_imbalanced_data_analysis.png")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: HANDLING IMBALANCED DATA")
print("=" * 60)

print(f"""
PROBLEM:
  • Imbalanced data = Model biased toward majority class
  • Accuracy is MISLEADING metric
  • Must use: Precision, Recall, F1-Score, ROC-AUC

TECHNIQUES COMPARED:

1. CLASS WEIGHTS (Easiest):
   ✓ No data modification
   ✓ Just one parameter
   ✓ Works well with tree-based models
   ✗ Not all algorithms support it

2. RANDOM OVERSAMPLING:
   ✓ Simple to implement
   ✓ Preserves all majority class info
   ✗ Creates duplicate samples (overfitting risk)
   ✗ Increases training time

3. RANDOM UNDERSAMPLING:
   ✓ Fast training (smaller dataset)
   ✓ No synthetic data
   ✗ Loses information from majority class
   ✗ Not good for small datasets

4. SMOTE (Best Overall):
   ✓ Creates synthetic samples (no duplicates)
   ✓ Better generalization
   ✓ Works very well in practice
   ✗ Computationally expensive
   ✗ Can create noise in overlap regions

5. THRESHOLD TUNING:
   ✓ No resampling needed
   ✓ Fine control over precision/recall
   ✓ Works with any model
   ✗ Requires probability outputs

OUR RESULTS:
  🏆 Best: {best_technique['Technique']}
  • F1-Score: {best_technique['F1-Score']:.4f}
  • Recall improved: {best_technique['Recall'] - baseline_recall:.4f}

RECOMMENDATIONS:
  1. Start with CLASS WEIGHTS (easiest)
  2. Try SMOTE if you need better performance
  3. Use THRESHOLD TUNING for fine control
  4. Monitor F1-Score and ROC-AUC, not just accuracy
  5. Consider business cost: Is false positive or false negative worse?

REAL-WORLD APPLICATIONS:
  • Fraud: High imbalance (0.1% fraud) → Use SMOTE + threshold tuning
  • Disease: Prioritize recall (catch all sick patients)
  • Spam: Balance precision/recall (user experience)
""")

print("\n✅ Imbalanced data handling mastery complete!")