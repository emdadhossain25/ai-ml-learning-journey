"""
Day 5: Seaborn - Statistical Visualizations
More powerful and beautiful than Matplotlib
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================
# 1. LOAD SAMPLE DATA
# ============================================

# Seaborn comes with built-in datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')

print("\nTips dataset preview:")
print(tips.head())

# ============================================
# 2. DISTRIBUTION PLOT
# ============================================

print("\n1. Creating distribution plot...")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(tips['total_bill'], kde=True, color='skyblue', bins=30)
plt.title('Total Bill Distribution', fontsize=14, fontweight='bold')

plt.subplot(1, 2, 2)
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, alpha=0.5)
plt.title('Total Bill by Time of Day', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/08_seaborn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/08_seaborn_distribution.png")

# ============================================
# 3. COUNT PLOT
# ============================================

print("2. Creating count plot...")

plt.figure(figsize=(10, 6))
sns.countplot(data=tips, x='day', hue='sex', palette='Set2')
plt.title('Restaurant Visits by Day and Gender', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.savefig('plots/09_count_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/09_count_plot.png")

# ============================================
# 4. BOX PLOT (Seaborn Style)
# ============================================

print("3. Creating box plot...")

plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', palette='Set1')
plt.title('Total Bill Distribution by Day and Gender', fontsize=14, fontweight='bold')

plt.savefig('plots/10_seaborn_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/10_seaborn_boxplot.png")

# ============================================
# 5. VIOLIN PLOT
# ============================================

print("4. Creating violin plot...")

plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill', hue='time', split=True, palette='muted')
plt.title('Total Bill Distribution (Violin Plot)', fontsize=14, fontweight='bold')

plt.savefig('plots/11_violin_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/11_violin_plot.png")

# ============================================
# 6. SCATTER PLOT WITH REGRESSION
# ============================================

print("5. Creating scatter plot with regression line...")

plt.figure(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'alpha':0.5}, color='darkblue')
plt.title('Tip Amount vs Total Bill (with trend line)', fontsize=14, fontweight='bold')
plt.xlabel('Total Bill ($)', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)

plt.savefig('plots/12_regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/12_regression_plot.png")

# ============================================
# 7. PAIR PLOT (Multiple relationships)
# ============================================

print("6. Creating pair plot (this takes a moment)...")

pair_plot = sns.pairplot(iris, hue='species', palette='bright', 
                         diag_kind='kde', height=2.5)
pair_plot.fig.suptitle('Iris Dataset - All Relationships', y=1.02, fontsize=16, fontweight='bold')

plt.savefig('plots/13_pair_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/13_pair_plot.png")

# ============================================
# 8. HEATMAP (Correlation Matrix)
# ============================================

print("7. Creating correlation heatmap...")

# Calculate correlations
correlation_matrix = tips[['total_bill', 'tip', 'size']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')

plt.savefig('plots/14_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/14_heatmap.png")

# ============================================
# 9. CATEGORICAL PLOT
# ============================================

print("8. Creating categorical plot...")

plt.figure(figsize=(12, 6))
sns.catplot(data=tips, x='day', y='total_bill', hue='sex', 
            kind='bar', height=6, aspect=1.5, palette='pastel')
plt.title('Average Bill by Day and Gender', fontsize=14, fontweight='bold')

plt.savefig('plots/15_categorical_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/15_categorical_plot.png")

print("\n" + "=" * 60)
print("✅ SEABORN PLOTS COMPLETE!")
print("=" * 60)
print("\nSeaborn gives you:")
print("  • Beautiful default styles")
print("  • Built-in statistical plots")
print("  • Easy categorical data visualization")
print("  • Automatic legends and colors")
print("  • Correlation heatmaps")
print("  • Regression lines")