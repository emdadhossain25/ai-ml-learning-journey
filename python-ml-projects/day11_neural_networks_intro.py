"""
Day 11: Neural Networks Fundamentals
Understanding the building blocks of deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("NEURAL NETWORKS FUNDAMENTALS")
print("=" * 60)

print(f"\n✅ TensorFlow version: {tf.__version__}")
print(f"✅ Keras version: {keras.__version__}")

# ============================================
# WHAT IS A NEURAL NETWORK?
# ============================================

print("\n1. UNDERSTANDING NEURAL NETWORKS")
print("-" * 60)

print("""
NEURAL NETWORK = Brain-Inspired Computing

BIOLOGICAL INSPIRATION:
  Brain Neuron → Artificial Neuron
  • Dendrites → Inputs
  • Cell Body → Computation
  • Axon → Output

ARTIFICIAL NEURON:
  1. Takes inputs (x₁, x₂, ..., xₙ)
  2. Multiplies by weights (w₁, w₂, ..., wₙ)
  3. Adds bias (b)
  4. Applies activation function
  5. Produces output

FORMULA:
  output = activation(Σ(xᵢ × wᵢ) + b)

COMPARISON TO TRADITIONAL ML:
┌─────────────────────────────────────────────────┐
│ Traditional ML    │  Deep Learning              │
├─────────────────────────────────────────────────┤
│ Manual features   │  Learns features            │
│ Shallow models    │  Deep (many layers)         │
│ Small data        │  Large data                 │
│ Fast training     │  Slow (needs GPU)           │
│ Interpretable     │  Black box                  │
│ Linear patterns   │  Complex patterns           │
└─────────────────────────────────────────────────┘

NETWORK STRUCTURE:
  Input Layer → Hidden Layer(s) → Output Layer
  
  Example: Digit Recognition (MNIST)
  Input: 28×28 = 784 pixels
  Hidden: Learn edges, curves, shapes
  Output: 10 neurons (digits 0-9)

KEY CONCEPTS:
  • Epoch: One pass through entire dataset
  • Batch: Subset of data (e.g., 32 samples)
  • Learning Rate: How fast model learns
  • Loss: How wrong predictions are
  • Optimizer: Algorithm to update weights
""")

# ============================================
# LOAD & PREPARE DATA
# ============================================

print("\n" + "=" * 60)
print("2. LOADING TITANIC DATASET")
print("=" * 60)

# Load Titanic data
df = pd.read_csv('data/titanic.csv')

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']

# Select features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
           'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare_Per_Person']

X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived'].values

print(f"✅ Data loaded:")
print(f"   Total passengers: {len(df)}")
print(f"   Features: {X.shape[1]}")
print(f"   Survival rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (CRITICAL for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData split:")
print(f"   Training: {len(X_train)} passengers")
print(f"   Test: {len(X_test)} passengers")

print(f"\n⚠️  Features MUST be scaled for neural networks!")
print(f"   Original range: {X_train.values.min():.2f} to {X_train.values.max():.2f}")
print(f"   Scaled range: {X_train_scaled.min():.2f} to {X_train_scaled.max():.2f}")

# ============================================
# BUILD SIMPLE NEURAL NETWORK
# ============================================

print("\n" + "=" * 60)
print("3. BUILDING NEURAL NETWORK")
print("=" * 60)

# Create model
model_simple = models.Sequential([
    # Input layer automatically created
    layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],), name='hidden_1'),
    layers.Dense(8, activation='relu', name='hidden_2'),
    layers.Dense(1, activation='sigmoid', name='output')
])

print("Simple Network Architecture:")
print("-" * 60)
model_simple.summary()

print(f"\nTotal parameters: {model_simple.count_params():,}")
print(f"\nLayer breakdown:")
print(f"  Hidden 1: {X_train_scaled.shape[1]} inputs × 16 neurons + 16 biases = {X_train_scaled.shape[1] * 16 + 16} params")
print(f"  Hidden 2: 16 inputs × 8 neurons + 8 biases = {16 * 8 + 8} params")
print(f"  Output: 8 inputs × 1 neuron + 1 bias = {8 * 1 + 1} params")

# ============================================
# COMPILE MODEL
# ============================================

print("\n" + "=" * 60)
print("4. COMPILING MODEL")
print("=" * 60)

model_simple.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled:")
print("   Optimizer: Adam (adaptive learning rate)")
print("   Loss: Binary Crossentropy (for 0/1 classification)")
print("   Metrics: Accuracy")

print("""
OPTIMIZER CHOICES:
  • SGD: Basic, slow but stable
  • Adam: Adaptive, fast, default choice ✓
  • RMSprop: Good for RNNs
  • Adagrad: Good for sparse data

LOSS FUNCTIONS:
  • Binary Crossentropy: Binary classification (0/1)
  • Categorical Crossentropy: Multi-class (one-hot)
  • MSE: Regression
  • MAE: Regression (robust to outliers)
""")

# ============================================
# TRAIN WITH CALLBACKS
# ============================================

print("\n" + "=" * 60)
print("5. TRAINING WITH EARLY STOPPING")
print("=" * 60)

# Create callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

print("Callbacks configured:")
print("  • EarlyStopping: Stop if no improvement for 10 epochs")
print("  • ReduceLROnPlateau: Reduce learning rate if stuck")

print("\nTraining for up to 100 epochs...")
print("(Early stopping may end training sooner)")
print()

history = model_simple.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✅ Training complete!")

# ============================================
# EVALUATE MODEL
# ============================================

print("\n" + "=" * 60)
print("6. EVALUATING MODEL")
print("=" * 60)

# Evaluate on test set
test_loss, test_acc = model_simple.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
y_pred_proba = model_simple.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ============================================
# BUILD DEEPER NETWORK
# ============================================

print("\n" + "=" * 60)
print("7. BUILDING DEEPER NETWORK")
print("=" * 60)

model_deep = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),  # Prevent overfitting
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(8, activation='relu'),
    
    layers.Dense(1, activation='sigmoid')
])

print("Deep Network Architecture:")
print("-" * 60)
model_deep.summary()

print(f"\n⚠️  Added Dropout layers:")
print(f"   Randomly drops 20-30% of neurons during training")
print(f"   Prevents overfitting by forcing network to learn robust features")

# Compile
model_deep.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("\nTraining deep network...")
history_deep = model_deep.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate
test_loss_deep, test_acc_deep = model_deep.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\n✅ Deep Network Results:")
print(f"   Test Accuracy: {test_acc_deep:.4f} ({test_acc_deep:.2%})")
print(f"   Improvement: {(test_acc_deep - test_acc)*100:.2f}%")

# ============================================
# COMPARISON
# ============================================

print("\n" + "=" * 60)
print("8. MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['Simple NN (2 layers)', 'Deep NN (4 layers)'],
    'Parameters': [model_simple.count_params(), model_deep.count_params()],
    'Test Accuracy': [test_acc, test_acc_deep],
    'Test Loss': [test_loss, test_loss_deep]
})

print("\n" + comparison.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("9. CREATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

fig.suptitle('Neural Network Training Analysis', fontsize=20, fontweight='bold')

# Plot 1: Training History - Simple Model Accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['accuracy'], linewidth=2.5, label='Training', color='blue')
ax1.plot(history.history['val_accuracy'], linewidth=2.5, label='Validation', color='red')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Simple Network - Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: Training History - Simple Model Loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['loss'], linewidth=2.5, label='Training', color='blue')
ax2.plot(history.history['val_loss'], linewidth=2.5, label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Simple Network - Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

# Plot 3: Deep Network Accuracy
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(history_deep.history['accuracy'], linewidth=2.5, label='Training', color='green')
ax3.plot(history_deep.history['val_accuracy'], linewidth=2.5, label='Validation', color='orange')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Deep Network - Accuracy', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3)

# Plot 4: Deep Network Loss
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(history_deep.history['loss'], linewidth=2.5, label='Training', color='green')
ax4.plot(history_deep.history['val_loss'], linewidth=2.5, label='Validation', color='orange')
ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax4.set_title('Deep Network - Loss', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# Plot 5: Confusion Matrix
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=['Died', 'Survived'],
           yticklabels=['Died', 'Survived'],
           annot_kws={'fontsize': 14})
ax5.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax5.set_title('Confusion Matrix (Deep Network)', fontsize=14, fontweight='bold')

# Plot 6: Architecture Comparison
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

architecture_text = f"""
╔══════════════════════════════════════════════════╗
║         NEURAL NETWORK ARCHITECTURES             ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  SIMPLE NETWORK:                                 ║
║    Input ({X_train_scaled.shape[1]} features)                          ║
║      ↓                                           ║
║    Dense (16 neurons, ReLU)                      ║
║      ↓                                           ║
║    Dense (8 neurons, ReLU)                       ║
║      ↓                                           ║
║    Dense (1 neuron, Sigmoid)                     ║
║      ↓                                           ║
║    Prediction (0 or 1)                           ║
║                                                  ║
║    Parameters: {model_simple.count_params():,}                               ║
║    Test Accuracy: {test_acc:.2%}                           ║
║                                                  ║
║  ─────────────────────────────────────────────   ║
║                                                  ║
║  DEEP NETWORK:                                   ║
║    Input ({X_train_scaled.shape[1]} features)                          ║
║      ↓                                           ║
║    Dense (64) + Dropout (30%)                    ║
║      ↓                                           ║
║    Dense (32) + Dropout (30%)                    ║
║      ↓                                           ║
║    Dense (16) + Dropout (20%)                    ║
║      ↓                                           ║
║    Dense (8)                                     ║
║      ↓                                           ║
║    Dense (1, Sigmoid)                            ║
║      ↓                                           ║
║    Prediction                                    ║
║                                                  ║
║    Parameters: {model_deep.count_params():,}                              ║
║    Test Accuracy: {test_acc_deep:.2%}                          ║
║                                                  ║
║  KEY INSIGHTS:                                   ║
║    • More layers ≠ always better                 ║
║    • Dropout prevents overfitting                ║
║    • Early stopping saves time                   ║
║    • Feature scaling is critical                 ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""

ax6.text(0.05, 0.5, architecture_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.savefig('plots/46_neural_network_intro.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/46_neural_network_intro.png")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("10. SAVING MODEL")
print("=" * 60)

# Save model
model_deep.save('models/titanic_neural_network.keras')
print("✅ Model saved: models/titanic_neural_network.keras")

# Save scaler
import joblib
joblib.dump(scaler, 'models/titanic_scaler.pkl')
print("✅ Scaler saved: models/titanic_scaler.pkl")

# Test loading
print("\nTesting model loading...")
loaded_model = keras.models.load_model('models/titanic_neural_network.keras')
loaded_scaler = joblib.load('models/titanic_scaler.pkl')

# Make prediction with loaded model
sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled, verbose=0)[0][0]

print(f"✅ Loaded model works!")
print(f"   Sample prediction: {prediction:.2%} survival probability")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: NEURAL NETWORKS")
print("=" * 60)

print(f"""
WHAT WE LEARNED:

1. NEURAL NETWORK BASICS:
   • Layers: Input → Hidden → Output
   • Neurons: Learn patterns from data
   • Activation: ReLU (hidden), Sigmoid (output)
   • Parameters: {model_deep.count_params():,} weights to learn

2. TRAINING PROCESS:
   • Forward pass: Input → Prediction
   • Calculate loss: How wrong?
   • Backward pass: Update weights
   • Repeat for many epochs

3. KEY CONCEPTS:
   • Epoch: One pass through all data
   • Batch: Subset of data (32 samples)
   • Early Stopping: Stop when no improvement
   • Dropout: Randomly disable neurons (prevent overfitting)

4. CRITICAL REQUIREMENTS:
   ⚠️  MUST scale features (StandardScaler)
   ⚠️  Sufficient data (100+ samples minimum)
   ⚠️  Choose right activation functions
   ⚠️  Monitor validation loss (not just training)

5. RESULTS:
   • Simple Network: {test_acc:.2%} accuracy
   • Deep Network: {test_acc_deep:.2%} accuracy
   • Parameters: Simple={model_simple.count_params():,}, Deep={model_deep.count_params():,}

6. ACTIVATION FUNCTIONS:
   • ReLU: f(x) = max(0, x) - Default choice
   • Sigmoid: f(x) = 1/(1+e⁻ˣ) - Binary output
   • Softmax: Multi-class probabilities
   • Tanh: f(x) = tanh(x) - Centered at 0

7. WHEN TO USE NEURAL NETWORKS:
   ✓ Large datasets (1000+ samples)
   ✓ Complex patterns
   ✓ Images, text, audio
   ✗ Small data → Use Random Forest
   ✗ Need interpretability → Use linear models

NEXT: Image classification with CNNs!
""")

print("\n✅ Neural Networks fundamentals complete!")