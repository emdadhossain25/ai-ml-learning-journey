"""
Day 5: Titanic Data Visualization
Bringing our Day 4 analysis to life with charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("TITANIC SURVIVAL VISUALIZATION")
print("=" * 60)

# Load data
df = pd.read_csv('data/titanic.csv')
df['Age'].fillna(df['Age'].median(),inplace=True)
print(f"Visualizing {len(df)} passengers ..\n")

print("1. Creating survival overview ...")
fig,axes = plt.subplots(1,2, figsize = (14, 5))

#Survival Count
survival_counts = df['Survived'].value_counts()
axes[0].pie(survival_counts, labels =['Died','Survived'], autopct = '%1.1f%%', 
            colors=['#ff6b6b', '#51cf66'], startangle=90, explode=(0.05, 0.05))
axes[0].set_title('Overall Survival Rate', fontsize=14, fontweight='bold')

#Survival by gender
survival_gender = df.groupby(['Sex','Survived']).size().unstack()
survival_gender.plot(kind='bar', ax = axes[1],color=['#ff6b6b', '#51cf66'], width=0.7)
axes[1].set_title('Survival by Gender', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gender', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_xticklabels(['Female', 'Male'], rotation=0)
axes[1].legend(['Died', 'Survived'])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/16_titanic_survival_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/16_titanic_survival_overview.png") 

# ============================================
# 2. SURVIVAL BY CLASS
# ============================================

print("2. Creating class analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Survival rate by class
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
axes[0].bar(['1st Class', '2nd Class', '3rd Class'], class_survival,
            color=['#ffd700', '#c0c0c0', '#cd7f32'], edgecolor='black', linewidth=1.5)
axes[0].set_title('Survival Rate by Class', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Survival Rate (%)', fontsize=12)
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, v in enumerate(class_survival):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# Count by class
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[1], palette=['#ff6b6b', '#51cf66'])
axes[1].set_title('Passenger Count by Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].legend(['Died', 'Survived'])

plt.tight_layout()
plt.savefig('plots/17_titanic_class_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/17_titanic_class_analysis.png")

# ============================================
# 3. AGE DISTRIBUTION
# ============================================

print("3. Creating age analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution histogram
axes[0, 0].hist([df[df['Survived']==1]['Age'], df[df['Survived']==0]['Age']],
                bins=30, label=['Survived', 'Died'], color=['#51cf66', '#ff6b6b'], alpha=0.7)
axes[0, 0].set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Age', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Age distribution violin plot
sns.violinplot(data=df, x='Survived', y='Age', ax=axes[0, 1], palette=['#ff6b6b', '#51cf66'])
axes[0, 1].set_title('Age Distribution (Violin Plot)', fontsize=14, fontweight='bold')
axes[0, 1].set_xticklabels(['Died', 'Survived'])

# Age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100],
                         labels=['Child', 'Teen', 'Adult', 'Senior'])
age_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
axes[1, 0].bar(range(len(age_survival)), age_survival,
               color=['#74c0fc', '#4dabf7', '#339af0', '#1c7ed6'])
axes[1, 0].set_xticks(range(len(age_survival)))
axes[1, 0].set_xticklabels(['Child', 'Teen', 'Adult', 'Senior'])
axes[1, 0].set_title('Survival Rate by Age Group', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1, 0].grid(axis='y', alpha=0.3)

# Add labels
for i, v in enumerate(age_survival):
    axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# Age by class
sns.boxplot(data=df, x='Pclass', y='Age', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Class', fontsize=12)

plt.tight_layout()
plt.savefig('plots/18_titanic_age_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/18_titanic_age_analysis.png")

# ============================================
# 4. FARE ANALYSIS
# ============================================

print("4. Creating fare analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fare distribution
sns.histplot(data=df, x='Fare', hue='Survived', bins=50, ax=axes[0],
             palette=['#ff6b6b', '#51cf66'], alpha=0.6)
axes[0].set_title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 300)
axes[0].legend(['Died', 'Survived'])

# Fare by class
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived', ax=axes[1],
            palette=['#ff6b6b', '#51cf66'])
axes[1].set_title('Fare by Class and Survival', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 300)
axes[1].legend(['Died', 'Survived'])

plt.tight_layout()
plt.savefig('plots/19_titanic_fare_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/19_titanic_fare_analysis.png")

# ============================================
# 5. FAMILY SIZE ANALYSIS
# ============================================

print("5. Creating family analysis...")

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Family size distribution
family_survival = df.groupby('FamilySize')['Survived'].agg(['mean', 'count'])
axes[0].bar(family_survival.index, family_survival['mean'] * 100,
            color='skyblue', edgecolor='black')
axes[0].set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Family Size', fontsize=12)
axes[0].set_ylabel('Survival Rate (%)', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Alone vs with family
df['Alone'] = (df['FamilySize'] == 1).astype(int)
alone_survival = df.groupby('Alone')['Survived'].mean() * 100
axes[1].bar(['With Family', 'Alone'], alone_survival,
            color=['#51cf66', '#ff6b6b'], edgecolor='black', linewidth=1.5)
axes[1].set_title('Survival: Alone vs With Family', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(alone_survival):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/20_titanic_family_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/20_titanic_family_analysis.png")

# ============================================
# 6. CORRELATION HEATMAP
# ============================================

print("6. Creating correlation heatmap...")

# Prepare numeric data
numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
correlation = df[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Titanic Features Correlation Matrix', fontsize=16, fontweight='bold')

plt.savefig('plots/21_titanic_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/21_titanic_correlation.png")

print("\n" + "=" * 60)
print("✅ TITANIC VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("\nCreated 6 comprehensive visualizations:")
print("  1. Survival overview")
print("  2. Class analysis")
print("  3. Age patterns")
print("  4. Fare distribution")
print("  5. Family size impact")
print("  6. Feature correlations")
print("\nYour data now tells a visual story!")