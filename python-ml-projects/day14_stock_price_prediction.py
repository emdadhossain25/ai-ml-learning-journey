"""
Day 14: Stock Price Prediction with LSTM
Predicting stock prices using real market data
"""
import joblib 
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
print("STOCK PRICE PREDICTION WITH LSTM")
print("=" * 60)

# ============================================
# DISCLAIMER
# ============================================

print("\n⚠️  IMPORTANT DISCLAIMER")
print("-" * 60)
print("""
STOCK MARKET PREDICTION:
  • Extremely difficult (markets are complex)
  • Past performance ≠ future results
  • Many factors beyond price history
  • News, politics, economics, psychology
  
THIS IS FOR EDUCATIONAL PURPOSES ONLY!
  • Learn LSTM techniques
  • Understand time series
  • NOT financial advice
  • DO NOT use for real trading
  
Professional traders use:
  • Fundamental analysis
  • News sentiment
  • Economic indicators
  • Domain expertise
  • Risk management
  
Our approach: Learn the technique, not beat the market!
""")

# ============================================
# CREATE SYNTHETIC STOCK DATA
# ============================================

print("\n" + "=" * 60)
print("1. CREATING SYNTHETIC STOCK DATA")
print("=" * 60)

print("Generating realistic stock price data...")

# Generate synthetic stock-like data
np.random.seed(42)
days = 1000

# Start price
price = 100.0
prices = [price]

# Random walk with drift (realistic stock behavior)
for _ in range(days - 1):
    # Daily return: small upward drift + randomness
    daily_return = np.random.normal(0.0005, 0.02)  # 0.05% drift, 2% volatility
    price = price * (1 + daily_return)
    prices.append(price)

prices = np.array(prices)

# Create DataFrame
dates = pd.date_range('2022-01-01', periods=days, freq='D')
df = pd.DataFrame({
    'Date': dates,
    'Close': prices
})

# Add technical indicators
df['MA7'] = df['Close'].rolling(window=7).mean()  # 7-day moving average
df['MA30'] = df['Close'].rolling(window=30).mean()  # 30-day moving average
df['Volume'] = np.random.randint(1000000, 5000000, size=days)  # Random volume

print(f"✅ Stock data created:")
print(f"   Days: {len(df)}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
print(f"   Mean price: ${df['Close'].mean():.2f}")

print(f"\nFirst 5 days:")
print(df.head())

print(f"\nLast 5 days:")
print(df.tail())

# ============================================
# EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Statistics
print(f"Price Statistics:")
print(f"  Min: ${df['Close'].min():.2f}")
print(f"  Max: ${df['Close'].max():.2f}")
print(f"  Mean: ${df['Close'].mean():.2f}")
print(f"  Std: ${df['Close'].std():.2f}")

# Daily returns
df['Daily_Return'] = df['Close'].pct_change()
print(f"\nDaily Return Statistics:")
print(f"  Mean: {df['Daily_Return'].mean():.4f} ({df['Daily_Return'].mean()*100:.2f}%)")
print(f"  Std: {df['Daily_Return'].std():.4f} ({df['Daily_Return'].std()*100:.2f}%)")
print(f"  Min: {df['Daily_Return'].min():.4f}")
print(f"  Max: {df['Daily_Return'].max():.4f}")

# Volatility (30-day rolling std of returns)
df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

# ============================================
# PREPARE DATA FOR LSTM
# ============================================

print("\n" + "=" * 60)
print("3. PREPARING DATA FOR LSTM")
print("=" * 60)

# Use only closing price for prediction
data = df['Close'].values

# Normalize (CRITICAL!)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

print(f"✅ Data normalized:")
print(f"   Original range: ${data.min():.2f} to ${data.max():.2f}")
print(f"   Scaled range: {data_scaled.min():.4f} to {data_scaled.max():.4f}")

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Use 60 days to predict next day (common in finance)
SEQ_LENGTH = 60

X, y = create_sequences(data_scaled, SEQ_LENGTH)

print(f"\n✅ Sequences created:")
print(f"   Sequence length: {SEQ_LENGTH} days")
print(f"   Total sequences: {len(X)}")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split (80/20, chronological!)
train_size = int(0.8 * len(X))

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print(f"\nTrain/test split:")
print(f"   Training: {len(X_train)} sequences")
print(f"   Test: {len(X_test)} sequences")

# ============================================
# BUILD LSTM MODEL
# ============================================

print("\n" + "=" * 60)
print("4. BUILDING LSTM MODEL")
print("=" * 60)

model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    layers.Dropout(0.2),
    
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    
    layers.LSTM(50),
    layers.Dropout(0.2),
    
    layers.Dense(25, activation='relu'),
    layers.Dense(1)
], name='StockLSTM')

print("Stock Price LSTM Architecture:")
print("-" * 60)
model.summary()

print(f"\nArchitecture highlights:")
print(f"  • 3 LSTM layers (50 units each)")
print(f"  • Dropout 20% (prevent overfitting)")
print(f"  • Dense layer (25 units)")
print(f"  • Output: Next day's price")
print(f"  • Total parameters: {model.count_params():,}")

# ============================================
# COMPILE & TRAIN
# ============================================

print("\n" + "=" * 60)
print("5. TRAINING MODEL")
print("=" * 60)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

print("Training stock prediction model...")
print("This may take 3-5 minutes...\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print(f"\n✅ Training complete!")
print(f"   Epochs trained: {len(history.history['loss'])}")

# ============================================
# EVALUATE & PREDICT
# ============================================

print("\n" + "=" * 60)
print("6. MAKING PREDICTIONS")
print("=" * 60)

# Predictions
train_predict = model.predict(X_train, verbose=0)
test_predict = model.predict(X_test, verbose=0)

# Inverse transform to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
train_mae = mean_absolute_error(y_train_actual, train_predict)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
train_r2 = r2_score(y_train_actual, train_predict)

test_mae = mean_absolute_error(y_test_actual, test_predict)
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
test_r2 = r2_score(y_test_actual, test_predict)

print(f"Training Set Performance:")
print(f"  MAE: ${train_mae:.2f}")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  R²: {train_r2:.4f}")

print(f"\nTest Set Performance:")
print(f"  MAE: ${test_mae:.2f}")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  R²: {test_r2:.4f}")

# Percentage error
mean_price = df['Close'].mean()
test_mape = (test_mae / mean_price) * 100

print(f"\nTest Set Error Percentage:")
print(f"  Mean Absolute Percentage Error: {test_mape:.2f}%")

# ============================================
# FUTURE PREDICTIONS
# ============================================

print("\n" + "=" * 60)
print("7. PREDICTING NEXT 30 DAYS")
print("=" * 60)

print("Generating future predictions (next 30 days)...")

# Start with last 60 days
last_sequence = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

future_predictions = []

for _ in range(30):
    # Predict next day
    next_pred = model.predict(last_sequence, verbose=0)
    future_predictions.append(next_pred[0, 0])
    
    # Update sequence (remove first, add prediction)
    last_sequence = np.append(last_sequence[:, 1:, :], 
                              next_pred.reshape(1, 1, 1), axis=1)

# Inverse transform
future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

print(f"✅ Future predictions generated:")
print(f"   Current price: ${data[-1]:.2f}")
print(f"   Predicted price (Day 30): ${future_predictions[-1][0]:.2f}")
print(f"   Change: ${future_predictions[-1][0] - data[-1]:.2f} "
      f"({((future_predictions[-1][0] / data[-1]) - 1) * 100:.2f}%)")

# ============================================
# TRADING SIGNALS
# ============================================

print("\n" + "=" * 60)
print("8. GENERATING TRADING SIGNALS")
print("=" * 60)

print("""
SIMPLE TRADING STRATEGY:
  • If predicted price > current price by 2%: BUY signal
  • If predicted price < current price by 2%: SELL signal
  • Otherwise: HOLD

⚠️  DISCLAIMER: This is for educational purposes only!
    DO NOT use this for actual trading!
""")

# Calculate signals
signals = []
for i in range(len(test_predict) - 1):
    current = y_test_actual[i][0]
    predicted = test_predict[i + 1][0]
    change_pct = ((predicted - current) / current) * 100
    
    if change_pct > 2:
        signal = 'BUY'
    elif change_pct < -2:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    signals.append(signal)

buy_count = signals.count('BUY')
sell_count = signals.count('SELL')
hold_count = signals.count('HOLD')

print(f"\nSignals generated on test set:")
print(f"  BUY: {buy_count} ({buy_count/len(signals)*100:.1f}%)")
print(f"  SELL: {sell_count} ({sell_count/len(signals)*100:.1f}%)")
print(f"  HOLD: {hold_count} ({hold_count/len(signals)*100:.1f}%)")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("9. CREATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.3)

fig.suptitle('Stock Price Prediction with LSTM', fontsize=20, fontweight='bold')

# Plot 1: Original Stock Price
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Date'], df['Close'], linewidth=1.5, color='blue', label='Close Price')
ax1.plot(df['Date'], df['MA7'], linewidth=1.5, color='orange', 
        alpha=0.7, label='7-Day MA')
ax1.plot(df['Date'], df['MA30'], linewidth=1.5, color='red',
        alpha=0.7, label='30-Day MA')
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.set_title('Stock Price History with Moving Averages', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: Training History - Loss
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history.history['loss'], linewidth=2.5, label='Training', color='blue')
ax2.plot(history.history['val_loss'], linewidth=2.5, label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax2.set_title('Model Training Loss', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Training History - MAE
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(history.history['mae'], linewidth=2.5, label='Training', color='green')
ax3.plot(history.history['val_mae'], linewidth=2.5, label='Validation', color='orange')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax3.set_title('Model Training MAE', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Predictions vs Actual (Training)
ax4 = fig.add_subplot(gs[2, 0])
train_time = range(len(train_predict))
ax4.plot(train_time, y_train_actual, linewidth=2, label='Actual', 
        color='blue', alpha=0.7)
ax4.plot(train_time, train_predict, linewidth=2, label='Predicted',
        color='red', alpha=0.6, linestyle='--')
ax4.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax4.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax4.set_title(f'Training Set Predictions (MAE: ${train_mae:.2f})',
             fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# Plot 5: Predictions vs Actual (Test)
ax5 = fig.add_subplot(gs[2, 1])
test_time = range(len(test_predict))
ax5.plot(test_time, y_test_actual, linewidth=2.5, label='Actual',
        color='blue', alpha=0.7)
ax5.plot(test_time, test_predict, linewidth=2.5, label='Predicted',
        color='red', alpha=0.6, linestyle='--')
ax5.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax5.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax5.set_title(f'Test Set Predictions (MAE: ${test_mae:.2f})',
             fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# Plot 6: Prediction Error Distribution
ax6 = fig.add_subplot(gs[3, 0])
test_errors = y_test_actual.flatten() - test_predict.flatten()
ax6.hist(test_errors, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Prediction Error ($)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Test Set Error Distribution', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Future Predictions
ax7 = fig.add_subplot(gs[3, 1])
# Last 60 days actual
historical = data[-60:]
historical_dates = range(60)
# Future 30 days
future_dates = range(60, 90)

ax7.plot(historical_dates, historical, linewidth=2.5, 
        label='Historical', color='blue')
ax7.plot(future_dates, future_predictions, linewidth=2.5,
        label='Future Predictions', color='red', linestyle='--')
ax7.axvline(x=59, color='green', linestyle='--', linewidth=2,
           label='Today')
ax7.set_xlabel('Days', fontsize=11, fontweight='bold')
ax7.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax7.set_title('Next 30 Days Forecast', fontsize=13, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(alpha=0.3)

# Plot 8: Performance Summary
ax8 = fig.add_subplot(gs[4, :])
ax8.axis('off')

summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    STOCK PRICE PREDICTION SUMMARY                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  MODEL ARCHITECTURE:                                                          ║
║    • 3 LSTM layers (50 units each)                                           ║
║    • Dropout: 20% after each LSTM                                            ║
║    • Dense layer: 25 units                                                   ║
║    • Total parameters: {model.count_params():,}                                          ║
║                                                                               ║
║  TRAINING PERFORMANCE:                                                        ║
║    • MAE: ${train_mae:.2f}                                                              ║
║    • RMSE: ${train_rmse:.2f}                                                            ║
║    • R² Score: {train_r2:.4f}                                                        ║
║                                                                               ║
║  TEST PERFORMANCE:                                                            ║
║    • MAE: ${test_mae:.2f}                                                               ║
║    • RMSE: ${test_rmse:.2f}                                                             ║
║    • R² Score: {test_r2:.4f}                                                         ║
║    • MAPE: {test_mape:.2f}%                                                              ║
║                                                                               ║
║  FORECAST (Next 30 Days):                                                     ║
║    • Current Price: ${data[-1]:.2f}                                                     ║
║    • Predicted (Day 30): ${future_predictions[-1][0]:.2f}                              ║
║    • Expected Change: {((future_predictions[-1][0] / data[-1]) - 1) * 100:+.2f}%                                          ║
║                                                                               ║
║  TRADING SIGNALS (Test Set):                                                  ║
║    • BUY signals: {buy_count} ({buy_count/len(signals)*100:.1f}%)                                                ║
║    • SELL signals: {sell_count} ({sell_count/len(signals)*100:.1f}%)                                               ║
║    • HOLD signals: {hold_count} ({hold_count/len(signals)*100:.1f}%)                                              ║
║                                                                               ║
║  ⚠️  IMPORTANT DISCLAIMER:                                                    ║
║    • This is for EDUCATIONAL PURPOSES ONLY                                   ║
║    • DO NOT use for actual trading decisions                                 ║
║    • Stock markets are influenced by countless factors                       ║
║    • Past performance does not guarantee future results                      ║
║    • Consult financial advisors for real investments                         ║
║                                                                               ║
║  LESSONS LEARNED:                                                             ║
║    • LSTM can capture temporal patterns                                      ║
║    • Price prediction is extremely challenging                               ║
║    • Many external factors affect stock prices                               ║
║    • Model shows the technique, not market-beating strategy                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

ax8.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.savefig('plots/54_stock_price_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/54_stock_price_prediction.png")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("10. SAVING MODEL")
print("=" * 60)

model.save('models/stock_price_lstm.keras')
joblib.dump(scaler, 'models/stock_price_scaler.pkl')

print("✅ Model saved: models/stock_price_lstm.keras")
print("✅ Scaler saved: models/stock_price_scaler.pkl")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: STOCK PREDICTION")
print("=" * 60)

print(f"""
WHAT WE LEARNED:

1. LSTM FOR STOCKS:
   • Can learn temporal patterns
   • Test MAE: ${test_mae:.2f} ({test_mape:.2f}% error)
   • R² Score: {test_r2:.4f}
   • Shows promise but not perfect

2. CHALLENGES:
   • Stock markets are chaotic
   • Influenced by news, politics, psychology
   • Past patterns don't guarantee future
   • "Random walk" hypothesis

3. ARCHITECTURE:
   • 3 LSTM layers (deep network)
   • Dropout prevents overfitting
   • 60-day lookback window
   • {model.count_params():,} parameters

4. REALISTIC EXPECTATIONS:
   • Model learns patterns, not fundamentals
   • Cannot predict black swan events
   • Cannot factor in breaking news
   • Works on historical patterns only

5. PROFESSIONAL TRADING:
   • Uses fundamental analysis
   • Monitors news and sentiment
   • Risk management crucial
   • Diversification essential

6. WHAT MODEL LEARNED:
   • Trend following
   • Moving average patterns
   • Mean reversion tendencies
   • Short-term momentum

7. LIMITATIONS:
   • Cannot predict market crashes
   • Cannot factor earnings reports
   • Cannot account for economic shocks
   • Overfits to historical patterns

REAL-WORLD APPLICATIONS:
  • Trend analysis (not absolute predictions)
  • Risk assessment
  • Portfolio optimization (with other factors)
  • Academic research
  • Feature in larger trading systems

ETHICAL CONSIDERATIONS:
  ⚠️  Never present this as "guaranteed profit"
  ⚠️  Always include risk disclaimers
  ⚠️  Encourage responsible investing
  ⚠️  Don't prey on financial desperation

BOTTOM LINE:
  LSTM is a powerful tool for sequence learning.
  Stock prediction is valuable for learning LSTM.
  But markets are too complex for simple price-based models.
  
  Use this to LEARN, not to GET RICH! 🎓

NEXT: Weather forecasting (more predictable!)
""")

print("\n✅ Stock price prediction complete!")
print("\n⚠️  Remember: This is educational only! Don't use for real trading!")