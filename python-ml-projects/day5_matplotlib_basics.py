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