"""
Day 14: LSTM vs Traditional ML for Time Series
Comparing deep learning with classical approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time as time_module  # Renamed to avoid conflict
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LSTM vs TRADITIONAL ML - COMPARISON")
print("=" * 60)

# ============================================
# CREATE DATA
# ============================================

print("\n1. CREATING TIME SERIES DATA")
print("-" * 60)

np.random.seed(42)
time_steps = 1000

time_array = np.arange(time_steps)  # Changed from 'time' to 'time_array'
trend = 0.02 * time_array + 10
seasonality = 5 * np.sin(2 * np.pi * time_array / 50)
noise = np.random.normal(0, 1, time_steps)
series = trend + seasonality + noise

print(f"✅ Data created: {len(series)} time steps")

# ============================================
# PREPARE DATA FOR DIFFERENT MODELS
# ============================================

print("\n" + "=" * 60)
print("2. PREPARING DATA")
print("=" * 60)

SEQ_LENGTH = 10

# For LSTM
def create_sequences_lstm(data, seq_length):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)
    
    return X, y, scaler

# For traditional ML
def create_sequences_ml(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)

# Prepare data
X_lstm, y_lstm, scaler = create_sequences_lstm(series, SEQ_LENGTH)
X_ml, y_ml = create_sequences_ml(series, SEQ_LENGTH)

# Split
train_size = int(0.8 * len(X_lstm))

X_lstm_train = X_lstm[:train_size]
X_lstm_test = X_lstm[train_size:]
y_lstm_train = y_lstm[:train_size]
y_lstm_test = y_lstm[train_size:]

X_ml_train = X_ml[:train_size]
X_ml_test = X_ml[train_size:]
y_ml_train = y_ml[:train_size]
y_ml_test = y_ml[train_size:]

print(f"✅ Data prepared for all models")
print(f"   Training: {train_size} samples")
print(f"   Test: {len(X_lstm_test)} samples")

# ============================================
# MODEL 1: LSTM
# ============================================

print("\n" + "=" * 60)
print("3. TRAINING LSTM")
print("=" * 60)

lstm_model = models.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

print("Training LSTM...")
start_time = time_module.time()  # Changed from time.time()

lstm_model.fit(X_lstm_train, y_lstm_train, 
              epochs=20, batch_size=32, verbose=0)

lstm_time = time_module.time() - start_time  # Changed from time.time()

lstm_pred = lstm_model.predict(X_lstm_test, verbose=0)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_lstm_test_orig = scaler.inverse_transform(y_lstm_test.reshape(-1, 1))

lstm_mae = mean_absolute_error(y_lstm_test_orig, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_lstm_test_orig, lstm_pred))
lstm_r2 = r2_score(y_lstm_test_orig, lstm_pred)

print(f"✅ LSTM Results:")
print(f"   MAE: {lstm_mae:.4f}")
print(f"   RMSE: {lstm_rmse:.4f}")
print(f"   R²: {lstm_r2:.4f}")
print(f"   Training time: {lstm_time:.2f}s")

# ============================================
# MODEL 2: RANDOM FOREST
# ============================================

print("\n" + "=" * 60)
print("4. TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training Random Forest...")
start_time = time_module.time()

rf_model.fit(X_ml_train, y_ml_train)

rf_time = time_module.time() - start_time

rf_pred = rf_model.predict(X_ml_test)

rf_mae = mean_absolute_error(y_ml_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_ml_test, rf_pred))
rf_r2 = r2_score(y_ml_test, rf_pred)

print(f"✅ Random Forest Results:")
print(f"   MAE: {rf_mae:.4f}")
print(f"   RMSE: {rf_rmse:.4f}")
print(f"   R²: {rf_r2:.4f}")
print(f"   Training time: {rf_time:.2f}s")

# ============================================
# MODEL 3: LINEAR REGRESSION
# ============================================

print("\n" + "=" * 60)
print("5. TRAINING LINEAR REGRESSION")
print("=" * 60)

lr_model = LinearRegression()

print("Training Linear Regression...")
start_time = time_module.time()

lr_model.fit(X_ml_train, y_ml_train)

lr_time = time_module.time() - start_time

lr_pred = lr_model.predict(X_ml_test)

lr_mae = mean_absolute_error(y_ml_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_ml_test, lr_pred))
lr_r2 = r2_score(y_ml_test, lr_pred)

print(f"✅ Linear Regression Results:")
print(f"   MAE: {lr_mae:.4f}")
print(f"   RMSE: {lr_rmse:.4f}")
print(f"   R²: {lr_r2:.4f}")
print(f"   Training time: {lr_time:.2f}s")

# ============================================
# COMPARISON
# ============================================

print("\n" + "=" * 60)
print("6. MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['LSTM', 'Random Forest', 'Linear Regression'],
    'MAE': [lstm_mae, rf_mae, lr_mae],
    'RMSE': [lstm_rmse, rf_rmse, lr_rmse],
    'R²': [lstm_r2, rf_r2, lr_r2],
    'Training Time (s)': [lstm_time, rf_time, lr_time]
})

print("\n" + comparison.to_string(index=False))

best_model = comparison.loc[comparison['MAE'].idxmin(), 'Model']
fastest_model = comparison.loc[comparison['Training Time (s)'].idxmin(), 'Model']

print(f"\n🏆 Best Accuracy: {best_model}")
print(f"⚡ Fastest Training: {fastest_model}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("7. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LSTM vs Traditional ML for Time Series', fontsize=18, fontweight='bold')

# Plot 1: Predictions Comparison
ax1 = axes[0, 0]
test_time_range = range(len(y_ml_test))
ax1.plot(test_time_range, y_ml_test, linewidth=2.5, label='Actual', color='blue', alpha=0.7)
ax1.plot(test_time_range, lstm_pred, linewidth=2, label='LSTM', color='red', alpha=0.6, linestyle='--')
ax1.plot(test_time_range, rf_pred, linewidth=2, label='Random Forest', color='green', alpha=0.6, linestyle='--')
ax1.plot(test_time_range, lr_pred, linewidth=2, label='Linear Reg', color='orange', alpha=0.6, linestyle='--')
ax1.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
ax1.set_title('Model Predictions Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: MAE Comparison
ax2 = axes[0, 1]
models = comparison['Model']
maes = comparison['MAE']
colors = ['lightcoral', 'lightgreen', 'lightblue']

bars = ax2.bar(models, maes, color=colors, edgecolor='black', linewidth=2)
ax2.set_ylabel('MAE (Lower is Better)', fontsize=11, fontweight='bold')
ax2.set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, mae in zip(bars, maes):
    ax2.text(bar.get_x() + bar.get_width()/2, mae + 0.01,
            f'{mae:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: Training Time Comparison
ax3 = axes[1, 0]
times = comparison['Training Time (s)']
bars = ax3.bar(models, times, color=colors, edgecolor='black', linewidth=2)
ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar, t in zip(bars, times):
    ax3.text(bar.get_x() + bar.get_width()/2, t + 0.1,
            f'{t:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Summary Table
ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
╔═══════════════════════════════════════════════╗
║      WHEN TO USE EACH APPROACH?               ║
╠═══════════════════════════════════════════════╣
║                                               ║
║  LSTM (Deep Learning):                        ║
║    ✓ Long sequences (100+ steps)              ║
║    ✓ Complex temporal patterns                ║
║    ✓ Multiple features (multivariate)         ║
║    ✓ Large datasets (10K+ samples)            ║
║    ✗ Small data (<1K samples)                 ║
║    ✗ Need fast training                       ║
║    ✗ Need interpretability                    ║
║                                               ║
║  Random Forest:                               ║
║    ✓ Medium-sized data (1K-100K)              ║
║    ✓ Need feature importance                  ║
║    ✓ Fast training required                   ║
║    ✓ Good baseline model                      ║
║    ✗ Very long sequences                      ║
║    ✗ Complex temporal dependencies            ║
║                                               ║
║  Linear Regression:                           ║
║    ✓ Simple trends                            ║
║    ✓ Need interpretability                    ║
║    ✓ Very fast training/prediction            ║
║    ✓ Baseline comparison                      ║
║    ✗ Complex patterns                         ║
║    ✗ Non-linear relationships                 ║
║                                               ║
║  RESULTS:                                     ║
║    Best Accuracy: {best_model:20s}     ║
║    Fastest: {fastest_model:20s}           ║
║                                               ║
║  RECOMMENDATION:                              ║
║    • Start with Random Forest                 ║
║    • Try LSTM if you have:                    ║
║      - Large data (10K+ samples)              ║
║      - Long sequences (50+ steps)             ║
║      - GPU available                          ║
║    • Use Linear Regression as baseline        ║
║                                               ║
╚═══════════════════════════════════════════════╝
"""

ax4.text(0.05, 0.5, summary, fontsize=9.5, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig('plots/56_lstm_vs_traditional.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/56_lstm_vs_traditional.png")

# ============================================
# FINAL VERDICT
# ============================================

print("\n" + "=" * 60)
print("FINAL VERDICT: WHEN TO USE WHAT?")
print("=" * 60)

print(f"""
COMPARISON RESULTS:
  LSTM:              MAE={lstm_mae:.3f}, Time={lstm_time:.1f}s
  Random Forest:     MAE={rf_mae:.3f}, Time={rf_time:.1f}s
  Linear Regression: MAE={lr_mae:.3f}, Time={lr_time:.1f}s

🏆 Winner: {best_model}
⚡ Fastest: {fastest_model}

KEY TAKEAWAY:
  For most time series problems, start with Random Forest!
  Only use LSTM if you have large data + GPU.

DAY 14 COMPLETE! 🎉
""")

print("\n✅ All comparisons complete!")