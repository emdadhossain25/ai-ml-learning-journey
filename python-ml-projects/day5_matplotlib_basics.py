"""
Day 5: Matplotlib Fundamentals
Creating professional visualizations for ML
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================
# 1. LINE PLOT
# ============================================

print("\n1. Creating a line plot...")

#simple data
x = np.linspace(0,10,100)
y = np.sin(x)


plt.figure(figsize=(10,6))
plt.plot(x,y,linewidth =2, color='blue', label ='sin(x)')
plt.plot(x,np.cos(x), linewidth=2, color ='red', linestyle ='--', label = 'cos(x)')
plt.title('Sine and Cosine Waves', fontsize =16, fontweight ='bold')
plt.title('X axis', fontsize =12)
plt.title('Y axis', fontsize =12)
plt.legend(fontsize=10)
plt.grid(True,alpha = 0.3)
plt.savefig('plots/01_line_plot.png', dpi =300, bbox_inches ='tight')
plt.close()
print("✅ Saved: plots/01_line_plot.png")



# ============================================
# 2. SCATTER PLOT
# ============================================

print("2. Creating a scatter plot...")

#Random data with correlation
np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + np.random.random(100) * 0.5

plt.figure(figsize=(10,6))
plt.scatter(x,y, alpha =0.6, c=y, cmap = 'viridis', s = 50, edgecolor = 'black')
plt.colorbar(label = 'Y Value')


plt.title('Scatter Plot with Correlation', fontsize = 16, fontweight = 'bold')
plt.xlabel('X values', fontsize = 12)
plt.ylabel('Y values', fontsize = 12)
plt.grid(True, alpha = 0.3)


plt.savefig('plots/02_scatter_plot.png', dpi = 300, bbox_inches ='tight')
plt.close()
print("✅ Saved: plots/02_scatter_plot.png")

# ============================================
# 3. BAR CHART
# ============================================

print("3. Creating a bar chart...")

categories = ['Python','Java', 'Javascript', 'C++', 'GO']
popularity = [85, 60, 75, 55, 45]
plt.figure(figsize =(10,6))
bars = plt.bar(categories, popularity, color = ['#3776ab', '#f89820', '#f7df1e', '#00599c', '#00add8'])


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
plt.title('Programming Language Popularity 2026', fontsize = 16, fontweight='bold')
plt.xlabel('Language', fontsize =12)
plt.ylabel('Popularity Score',fontsize =12)
plt.ylim(0,100)
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/03_bar_chart.png', dpi =300, bbox_inches = 'tight')
plt.close()
print("✅ Saved: plots/03_bar_chart.png")



# ============================================
# 4. HISTOGRAM
# ============================================

print("4. Creating a histogram...")

# Generate normal distribution
data = np.random.normal(100,15, 1000)
plt.figure(figsize=(10,6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of IQ Scores', fontsize=16, fontweight='bold')
plt.xlabel('IQ Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

plt.savefig('plots/04_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/04_histogram.png")


# ============================================
# 5. PIE CHART
# ============================================

print("5. Creating a pie chart...")

labels = ['1st Class', '2nd Class', '3rd Class']
sizes = [216, 184, 491]
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)  # Explode 1st slice

plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'})

plt.title('Titanic Passenger Class Distribution', fontsize=16, fontweight='bold')
plt.axis('equal')

plt.savefig('plots/05_pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/05_pie_chart.png")


# ============================================
# 6. SUBPLOTS - Multiple Charts
# ============================================

print("6. Creating subplots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multiple Visualizations Dashboard', fontsize=18, fontweight='bold')

# Plot 1: Line
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sine Wave')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
axes[0, 1].set_title('Random Scatter')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bar
axes[1, 0].bar(['A', 'B', 'C', 'D'], [10, 25, 15, 30], color='coral')
axes[1, 0].set_title('Category Comparison')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Histogram
axes[1, 1].hist(np.random.normal(0, 1, 1000), bins=30, color='lightgreen', edgecolor='black')
axes[1, 1].set_title('Normal Distribution')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/06_subplots_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/06_subplots_dashboard.png")
# ============================================
# 7. BOX PLOT
# ============================================

print("7. Creating a box plot...")

# Simulate test scores for 4 groups
data_groups = [np.random.normal(75, 10, 100),
               np.random.normal(80, 8, 100),
               np.random.normal(70, 12, 100),
               np.random.normal(85, 7, 100)]

plt.figure(figsize=(10, 6))
bp = plt.boxplot(data_groups, labels=['Class A', 'Class B', 'Class C', 'Class D'],
                 patch_artist=True, notch=True)

# Color boxes
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Test Score Distribution by Class', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.savefig('plots/07_box_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/07_box_plot.png")

print("\n" + "=" * 60)
print("✅ ALL PLOTS CREATED!")
print("=" * 60)
print("Check the 'plots' folder to see your visualizations!")
print("\nYou just learned:")
print("  • Line plots (trends over time)")
print("  • Scatter plots (relationships)")
print("  • Bar charts (comparisons)")
print("  • Histograms (distributions)")
print("  • Pie charts (proportions)")
print("  • Subplots (dashboards)")
print("  • Box plots (statistical summaries)")