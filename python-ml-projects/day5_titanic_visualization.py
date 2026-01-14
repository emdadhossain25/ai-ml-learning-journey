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
