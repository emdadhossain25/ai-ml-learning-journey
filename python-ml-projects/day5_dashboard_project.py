"""
Day 5: Ultimate Titanic Dashboard
Professional multi-panel visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Professional styling
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

print("=" * 60)
print("BUILDING COMPREHENSIVE DASHBOARD")
print("=" * 60)

# Load and prepare data
df = pd.read_csv('data/titanic.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100],
                         labels=['Child', 'Teen', 'Adult', 'Senior'])

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
fig.suptitle('TITANIC SURVIVAL ANALYSIS DASHBOARD', 
             fontsize=24, fontweight='bold', y=0.98)

# Define grid
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# ============================================
# PANEL 1: Overall Statistics
# ============================================
ax1 = fig.add_subplot(gs[0, 0])
total = len(df)
survived = df['Survived'].sum()
died = total - survived
survival_rate = (survived / total) * 100

ax1.text(0.5, 0.8, f'{total}', ha='center', fontsize=36, fontweight='bold')
ax1.text(0.5, 0.6, 'Total Passengers', ha='center', fontsize=14)
ax1.text(0.5, 0.4, f'{survival_rate:.1f}%', ha='center', fontsize=28, 
         fontweight='bold', color='green')
ax1.text(0.5, 0.2, 'Survived', ha='center', fontsize=14, color='green')
ax1.axis('off')
ax1.set_title('Overview', fontsize=14, fontweight='bold', pad=10)

# ============================================
# PANEL 2: Survival Pie Chart
# ============================================
ax2 = fig.add_subplot(gs[0, 1])
survival_counts = df['Survived'].value_counts()
ax2.pie(survival_counts, labels=['Died', 'Survived'], autopct='%1.1f%%',
        colors=['#ff6b6b', '#51cf66'], startangle=90, 
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Survival Distribution', fontsize=14, fontweight='bold')

# ============================================
# PANEL 3: Gender Survival
# ============================================
ax3 = fig.add_subplot(gs[0, 2:])
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
bars = ax3.bar(['Female', 'Male'], gender_survival, 
               color=['#ffc0cb', '#87ceeb'], edgecolor='black', linewidth=2)
ax3.set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
ax3.set_ylabel('Survival Rate (%)', fontsize=12)
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, gender_survival)):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 3, 
             f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold')

# ============================================
# PANEL 4: Class Analysis
# ============================================
ax4 = fig.add_subplot(gs[1, 0])
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
bars = ax4.barh(['3rd', '2nd', '1st'], class_survival, 
                color=['#cd7f32', '#c0c0c0', '#ffd700'], edgecolor='black')
ax4.set_title('Survival by Class', fontsize=14, fontweight='bold')
ax4.set_xlabel('Survival Rate (%)', fontsize=12)
ax4.set_xlim(0, 100)
ax4.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, class_survival):
    ax4.text(val + 2, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

# ============================================
# PANEL 5: Age Distribution
# ============================================
ax5 = fig.add_subplot(gs[1, 1:3])
ax5.hist([df[df['Survived']==1]['Age'], df[df['Survived']==0]['Age']],
         bins=30, label=['Survived', 'Died'], 
         color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black')
ax5.set_title('Age Distribution', fontsize=14, fontweight='bold')
ax5.set_xlabel('Age', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.legend(fontsize=11)
ax5.grid(axis='y', alpha=0.3)

# ============================================
# PANEL 6: Age Group Survival
# ============================================
ax6 = fig.add_subplot(gs[1, 3])
age_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
bars = ax6.bar(range(len(age_survival)), age_survival,
               color=['#74c0fc', '#4dabf7', '#339af0', '#1c7ed6'],
               edgecolor='black')
ax6.set_xticks(range(len(age_survival)))
ax6.set_xticklabels(['Child', 'Teen', 'Adult', 'Senior'], rotation=45)
ax6.set_title('Survival by Age Group', fontsize=14, fontweight='bold')
ax6.set_ylabel('Survival Rate (%)', fontsize=12)
ax6.set_ylim(0, 100)
ax6.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, age_survival)):
    ax6.text(i, val + 3, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# ============================================
# PANEL 7: Family Size Impact
# ============================================
ax7 = fig.add_subplot(gs[2, :2])
family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
ax7.plot(family_survival.index, family_survival, marker='o', 
         linewidth=3, markersize=10, color='#845ef7')
ax7.fill_between(family_survival.index, 0, family_survival, alpha=0.3, color='#845ef7')
ax7.set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
ax7.set_xlabel('Family Size', fontsize=12)
ax7.set_ylabel('Survival Rate (%)', fontsize=12)
ax7.set_ylim(0, 100)
ax7.grid(True, alpha=0.3)

# ============================================
# PANEL 8: Correlation Heatmap
# ============================================
ax8 = fig.add_subplot(gs[2, 2:])
numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation = df[numeric_features].corr()
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8}, 
            fmt='.2f', ax=ax8, annot_kws={'fontsize': 9})
ax8.set_title('Feature Correlations', fontsize=14, fontweight='bold')

# Add footer
fig.text(0.5, 0.02, 'Data Science Project by Emdad Hossain | Day 5 of 100 Days of Code', 
         ha='center', fontsize=12, style='italic')

plt.savefig('plots/22_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/22_comprehensive_dashboard.png")
plt.close()

print("\n" + "=" * 60)
print("🎉 DASHBOARD COMPLETE!")
print("=" * 60)
print("\nYou just created a professional data science dashboard!")
print("This is portfolio-worthy work!")