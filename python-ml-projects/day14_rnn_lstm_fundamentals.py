"""
Day 14: RNN & LSTM Fundamentals for Time Series
Understanding sequential data and recurrent networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RNN & LSTM FOR TIME SERIES")
print("=" * 60)

# ============================================
# WHAT ARE RNNs & LSTMs?
# ============================================

print("\n1. UNDERSTANDING SEQUENTIAL MODELS")
print("-" * 60)

print("""
TIME SERIES: Data points ordered in time
  • Stock prices
  • Weather patterns
  • Sales data
  • Website traffic
  • Your heart rate over time

WHY REGULAR NNs DON'T WORK:
  Regular NN: Each input is independent
  Time Series: Current value depends on previous values!
  
  Example: Tomorrow's temperature depends on today's

RECURRENT NEURAL NETWORK (RNN):
  • Has memory of previous inputs
  • Output depends on current + past inputs
  • "Recurrent" = loops back to itself

PROBLEM WITH RNNs:
  • Vanishing gradient problem
  • Can't remember long sequences
  • Forgets information from 50+ steps ago

LONG SHORT-TERM MEMORY (LSTM):
  • Special type of RNN
  • Has "gates" that control memory
  • Can remember long-term dependencies
  • Industry standard for sequences

LSTM GATES:
  1. FORGET GATE: What to forget from memory
  2. INPUT GATE: What new info to store
  3. OUTPUT GATE: What to output
  
  Think of it as: Brain deciding what to remember/forget

COMPARISON:
┌─────────────────────────────────────────────┐
│ CNN              │  RNN/LSTM                │
├─────────────────────────────────────────────┤
│ Spatial data     │  Sequential data         │
│ Images           │  Time series             │
│ 2D patterns      │  Temporal patterns       │
│ No memory        │  Has memory              │
└─────────────────────────────────────────────┘

APPLICATIONS:
  • Stock prediction
  • Weather forecasting
  • Language translation
  • Speech recognition
  • Video analysis
  • Anomaly detection
""")

# ============================================
# CREATE SYNTHETIC TIME SERIES
# ============================================

print("\n" + "=" * 60)
print("2. CREATING TIME SERIES DATA")
print("=" * 60)

print("Generating synthetic data with trend + seasonality + noise...")

# Time steps
np.random.seed(42)
time_steps = 1000

# Components
time = np.arange(time_steps)
trend = 0.02 * time + 10  # Linear trend
seasonality = 5 * np.sin(2 * np.pi * time / 50)  # Seasonal pattern
noise = np.random.normal(0, 1, time_steps)  # Random noise

# Combine
series = trend + seasonality + noise

print(f"✅ Time series created:")
print(f"   Length: {len(series)} time steps")
print(f"   Min: {series.min():.2f}")
print(f"   Max: {series.max():.2f}")
print(f"   Mean: {series.mean():.2f}")

# ============================================
# PREPARE DATA FOR LSTM
# ============================================

print("\n" + "=" * 60)
print("3. PREPARING DATA FOR LSTM")
print("=" * 60)

print("""
LSTM INPUT FORMAT: 3D tensor
  Shape: (samples, time_steps, features)
  
  Example: Predict tomorrow using last 50 days
    • samples: Number of sequences
    • time_steps: 50 (look back 50 days)
    • features: 1 (just the price)

SLIDING WINDOW:
  Day 1-50 → Predict Day 51
  Day 2-51 → Predict Day 52
  Day 3-52 → Predict Day 53
  ...
  
  This creates many training samples!
""")

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Input: seq_length previous values
        X.append(data[i:i + seq_length])
        # Output: next value
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)

# Normalize data (CRITICAL for neural networks!)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Sequence length (look back 50 steps)
SEQ_LENGTH = 50

# Create sequences
X, y = create_sequences(series_scaled, SEQ_LENGTH)

print(f"✅ Sequences created:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Explanation: {X.shape[0]} samples, each with {X.shape[1]} time steps")

# Reshape for LSTM (needs 3D: samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"   X reshaped: {X.shape}")
print(f"   Now 3D for LSTM input!")

# Train/test split (80/20, chronological order!)
train_size = int(0.8 * len(X))

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print(f"\n✅ Train/test split (chronological):")
print(f"   Training: {len(X_train)} sequences")
print(f"   Test: {len(X_test)} sequences")

# ============================================
# BUILD SIMPLE RNN
# ============================================

print("\n" + "=" * 60)
print("4. BUILDING SIMPLE RNN")
print("=" * 60)

model_rnn = models.Sequential([
    layers.SimpleRNN(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    layers.Dense(1)
], name='SimpleRNN')

print("Simple RNN Architecture:")
print("-" * 60)
model_rnn.summary()

model_rnn.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nTraining Simple RNN...")
history_rnn = model_rnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
rnn_loss = model_rnn.evaluate(X_test, y_test, verbose=0)
rnn_pred = model_rnn.predict(X_test, verbose=0)

# Inverse transform predictions
rnn_pred_orig = scaler.inverse_transform(rnn_pred)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

rnn_mae = mean_absolute_error(y_test_orig, rnn_pred_orig)
rnn_rmse = np.sqrt(mean_squared_error(y_test_orig, rnn_pred_orig))

print(f"\n✅ Simple RNN Results:")
print(f"   MAE: {rnn_mae:.4f}")
print(f"   RMSE: {rnn_rmse:.4f}")

# ============================================
# BUILD LSTM
# ============================================

print("\n" + "=" * 60)
print("5. BUILDING LSTM")
print("=" * 60)

model_lstm = models.Sequential([
    layers.LSTM(50, activation='relu', return_sequences=True, 
               input_shape=(SEQ_LENGTH, 1)),
    layers.LSTM(50, activation='relu'),
    layers.Dense(1)
], name='LSTM')

print("LSTM Architecture:")
print("-" * 60)
model_lstm.summary()

print(f"\nKey differences from RNN:")
print(f"  • LSTM cells instead of SimpleRNN")
print(f"  • return_sequences=True in first layer (stacked LSTMs)")
print(f"  • Better at long-term dependencies")

model_lstm.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nTraining LSTM (2 layers, 50 units each)...")
print("This may take 2-3 minutes...\n")

history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
lstm_loss = model_lstm.evaluate(X_test, y_test, verbose=0)
lstm_pred = model_lstm.predict(X_test, verbose=0)

# Inverse transform
lstm_pred_orig = scaler.inverse_transform(lstm_pred)

lstm_mae = mean_absolute_error(y_test_orig, lstm_pred_orig)
lstm_rmse = np.sqrt(mean_squared_error(y_test_orig, lstm_pred_orig))

print(f"\n✅ LSTM Results:")
print(f"   MAE: {lstm_mae:.4f}")
print(f"   RMSE: {lstm_rmse:.4f}")

# ============================================
# BUILD BIDIRECTIONAL LSTM
# ============================================

print("\n" + "=" * 60)
print("6. BUILDING BIDIRECTIONAL LSTM")
print("=" * 60)

print("""
BIDIRECTIONAL LSTM:
  • Processes sequence forward AND backward
  • Sees future context (when available)
  • Better for: Text analysis, completed sequences
  • Not ideal for: Real-time prediction (no future data)
  
  We'll build it for comparison!
""")

model_bilstm = models.Sequential([
    layers.Bidirectional(layers.LSTM(50, activation='relu', return_sequences=True),
                        input_shape=(SEQ_LENGTH, 1)),
    layers.Bidirectional(layers.LSTM(50, activation='relu')),
    layers.Dense(1)
], name='BiLSTM')

print("Bidirectional LSTM Architecture:")
print("-" * 60)
model_bilstm.summary()

model_bilstm.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nTraining Bidirectional LSTM...")

history_bilstm = model_bilstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
bilstm_loss = model_bilstm.evaluate(X_test, y_test, verbose=0)
bilstm_pred = model_bilstm.predict(X_test, verbose=0)

# Inverse transform
bilstm_pred_orig = scaler.inverse_transform(bilstm_pred)

bilstm_mae = mean_absolute_error(y_test_orig, bilstm_pred_orig)
bilstm_rmse = np.sqrt(mean_squared_error(y_test_orig, bilstm_pred_orig))

print(f"\n✅ Bidirectional LSTM Results:")
print(f"   MAE: {bilstm_mae:.4f}")
print(f"   RMSE: {bilstm_rmse:.4f}")

# ============================================
# MODEL COMPARISON
# ============================================

print("\n" + "=" * 60)
print("7. COMPARING ALL MODELS")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['Simple RNN', 'LSTM', 'Bidirectional LSTM'],
    'MAE': [rnn_mae, lstm_mae, bilstm_mae],
    'RMSE': [rnn_rmse, lstm_rmse, bilstm_rmse],
    'Parameters': [
        model_rnn.count_params(),
        model_lstm.count_params(),
        model_bilstm.count_params()
    ]
})

print("\n" + comparison.to_string(index=False))

best_model = comparison.loc[comparison['MAE'].idxmin(), 'Model']
print(f"\n🏆 Best Model: {best_model}")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("8. CREATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

fig.suptitle('RNN & LSTM for Time Series Forecasting', 
             fontsize=20, fontweight='bold')

# Plot 1: Original Time Series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(series, linewidth=1.5, color='blue', alpha=0.7)
ax1.axvline(x=train_size, color='red', linestyle='--', linewidth=2,
           label='Train/Test Split')
ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('Complete Time Series (Trend + Seasonality + Noise)', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: RNN Training History
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history_rnn.history['loss'], linewidth=2.5, label='Training', color='blue')
ax2.plot(history_rnn.history['val_loss'], linewidth=2.5, label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax2.set_title('Simple RNN - Training Loss', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: LSTM Training History
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(history_lstm.history['loss'], linewidth=2.5, label='Training', color='green')
ax3.plot(history_lstm.history['val_loss'], linewidth=2.5, label='Validation', color='orange')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax3.set_title('LSTM - Training Loss', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Predictions Comparison
ax4 = fig.add_subplot(gs[2, :])
test_time = np.arange(len(y_test_orig))
ax4.plot(test_time, y_test_orig, linewidth=2.5, label='Actual', 
        color='blue', alpha=0.7)
ax4.plot(test_time, rnn_pred_orig, linewidth=2, label='RNN', 
        color='red', alpha=0.6, linestyle='--')
ax4.plot(test_time, lstm_pred_orig, linewidth=2, label='LSTM',
        color='green', alpha=0.6, linestyle='--')
ax4.plot(test_time, bilstm_pred_orig, linewidth=2, label='BiLSTM',
        color='orange', alpha=0.6, linestyle='--')
ax4.set_xlabel('Test Time Step', fontsize=12, fontweight='bold')
ax4.set_ylabel('Value', fontsize=12, fontweight='bold')
ax4.set_title('Model Predictions vs Actual', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# Plot 5: Model Performance Bar Chart
ax5 = fig.add_subplot(gs[3, 0])
models = comparison['Model']
maes = comparison['MAE']
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = ax5.bar(models, maes, color=colors, edgecolor='black', linewidth=2)
ax5.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax5.set_title('Model Performance (Lower is Better)', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

for bar, mae in zip(bars, maes):
    ax5.text(bar.get_x() + bar.get_width()/2, mae + 0.05,
            f'{mae:.3f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

# Plot 6: LSTM Architecture Diagram
ax6 = fig.add_subplot(gs[3, 1])
ax6.axis('off')

lstm_diagram = f"""
╔════════════════════════════════════════════╗
║      LSTM ARCHITECTURE EXPLAINED           ║
╠════════════════════════════════════════════╣
║                                            ║
║  INPUT: (samples, 50 timesteps, 1 feature) ║
║    ↓                                       ║
║  LSTM LAYER 1 (50 units)                   ║
║    • Forget Gate: What to forget           ║
║    • Input Gate: What to remember          ║
║    • Output Gate: What to output           ║
║    • return_sequences=True                 ║
║    ↓                                       ║
║  LSTM LAYER 2 (50 units)                   ║
║    • Same gates                            ║
║    • Deeper understanding                  ║
║    ↓                                       ║
║  DENSE LAYER (1 unit)                      ║
║    • Final prediction                      ║
║    ↓                                       ║
║  OUTPUT: Next value prediction             ║
║                                            ║
║  COMPARISON RESULTS:                       ║
║    • Simple RNN: MAE={rnn_mae:.3f}              ║
║    • LSTM: MAE={lstm_mae:.3f}                   ║
║    • BiLSTM: MAE={bilstm_mae:.3f}               ║
║                                            ║
║  🏆 WINNER: {best_model}               ║
║                                            ║
║  WHY LSTM WORKS:                           ║
║    • Remembers long-term patterns          ║
║    • Gates control information flow        ║
║    • Solves vanishing gradient             ║
║    • Industry standard for sequences       ║
║                                            ║
╚════════════════════════════════════════════╝
"""

ax6.text(0.05, 0.5, lstm_diagram, fontsize=9.5, verticalalignment='center',
        family='monospace', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.savefig('plots/53_rnn_lstm_fundamentals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/53_rnn_lstm_fundamentals.png")

# ============================================
# SAVE BEST MODEL
# ============================================

print("\n" + "=" * 60)
print("9. SAVING BEST MODEL")
print("=" * 60)

# Save LSTM (usually best)
model_lstm.save('models/time_series_lstm.keras')
print("✅ Model saved: models/time_series_lstm.keras")

# Save scaler
import joblib
joblib.dump(scaler, 'models/time_series_scaler.pkl')
print("✅ Scaler saved: models/time_series_scaler.pkl")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: RNN & LSTM")
print("=" * 60)

print(f"""
WHAT WE LEARNED:

1. RNN vs LSTM:
   • RNN: Simple, fast, but forgets long sequences
   • LSTM: Complex, slower, remembers long-term
   • BiLSTM: Best when future context available

2. RESULTS:
   • Simple RNN: MAE={rnn_mae:.3f}
   • LSTM: MAE={lstm_mae:.3f}
   • BiLSTM: MAE={bilstm_mae:.3f}
   • Winner: {best_model}

3. DATA PREPARATION:
   • Sliding window: Past 50 → Predict next
   • 3D tensor: (samples, timesteps, features)
   • Normalization: CRITICAL for convergence
   • Chronological split: Never shuffle!

4. LSTM COMPONENTS:
   • Forget Gate: Remove irrelevant info
   • Input Gate: Add new information
   • Output Gate: Decide what to output
   • Cell State: Long-term memory

5. WHEN TO USE:
   ✓ Time series forecasting
   ✓ Sequential data
   ✓ Dependencies between time steps
   ✓ Stock prices, weather, sales
   
   ✗ Independent data points
   ✗ No temporal pattern
   ✗ Very short sequences (<10 steps)

6. HYPERPARAMETERS:
   • Sequence length: 50 (look back)
   • Units: 50 (LSTM neurons)
   • Layers: 2 (stacked for complexity)
   • Batch size: 32
   • Epochs: 20

7. CHALLENGES:
   • Vanishing/exploding gradients (LSTM solves this)
   • Requires normalization
   • Slow training (compared to CNN)
   • Need sufficient sequence length

NEXT: Real stock price prediction!
""")

print("\n✅ RNN & LSTM fundamentals complete!")