"""
Day 12: Convolutional Neural Networks (CNN) Fundamentals
Understanding how CNNs see images
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CONVOLUTIONAL NEURAL NETWORKS (CNN)")
print("=" * 60)

# ============================================
# WHAT IS A CNN?
# ============================================

print("\n1. UNDERSTANDING CNNs")
print("-" * 60)

print("""
CNN = Convolutional Neural Network
Specialized for processing grid-like data (images, video)

WHY CNNs FOR IMAGES?
┌─────────────────────────────────────────────────────┐
│ Regular Neural Network │  CNN                       │
├─────────────────────────────────────────────────────┤
│ Treats pixels as        │  Understands spatial       │
│ independent features    │  relationships             │
│                         │                            │
│ 28×28 image =           │  Learns local patterns     │
│ 784 separate numbers    │  (edges, shapes, objects)  │
│                         │                            │
│ Huge parameters         │  Shared weights            │
│ Overfits easily         │  Better generalization     │
└─────────────────────────────────────────────────────┘

HOW CNNs WORK:
  1. CONVOLUTION LAYER:
     • Sliding window (filter/kernel)
     • Detects edges, textures, patterns
     • Example: 3×3 filter slides across image
  
  2. POOLING LAYER:
     • Reduces spatial dimensions
     • MaxPooling: Takes maximum value in region
     • Makes network translation-invariant
  
  3. FULLY CONNECTED:
     • Traditional neural network
     • Makes final classification

ARCHITECTURE EXAMPLE:
  Input (28×28×1)
    ↓ Conv2D (32 filters, 3×3)
  26×26×32
    ↓ MaxPooling (2×2)
  13×13×32
    ↓ Conv2D (64 filters, 3×3)
  11×11×64
    ↓ MaxPooling (2×2)
  5×5×64
    ↓ Flatten
  1600 neurons
    ↓ Dense (128)
  128 neurons
    ↓ Dense (10)
  10 classes

KEY CONCEPTS:
  • Filter/Kernel: Small matrix that slides over image
  • Feature Map: Output of convolution
  • Stride: How many pixels to move filter
  • Padding: Add borders to preserve size
""")

# ============================================
# LOAD MNIST DATASET
# ============================================

print("\n" + "=" * 60)
print("2. LOADING MNIST DATASET")
print("=" * 60)

print("Downloading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n✅ MNIST loaded:")
print(f"   Training images: {X_train.shape[0]:,}")
print(f"   Test images: {X_test.shape[0]:,}")
print(f"   Image size: {X_train.shape[1]}×{X_train.shape[2]}")
print(f"   Classes: {len(np.unique(y_train))} (digits 0-9)")

# Show sample images
print(f"\nSample images and labels:")
for i in range(5):
    print(f"  Image {i}: Label = {y_train[i]}")

# ============================================
# PREPROCESS DATA
# ============================================

print("\n" + "=" * 60)
print("3. PREPROCESSING")
print("=" * 60)

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_onehot = keras.utils.to_categorical(y_train, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

print(f"✅ Preprocessing complete:")
print(f"   Training shape: {X_train.shape}")
print(f"   Test shape: {X_test.shape}")
print(f"   Pixel range: {X_train.min():.1f} to {X_train.max():.1f}")
print(f"   Labels: One-hot encoded (10 classes)")

print(f"\nExample one-hot encoding:")
print(f"  Digit {y_train[0]} → {y_train_onehot[0]}")

# ============================================
# BUILD CNN
# ============================================

print("\n" + "=" * 60)
print("4. BUILDING CNN ARCHITECTURE")
print("=" * 60)

model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    
    # Third Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
    
    # Flatten and Dense Layers
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout'),
    layers.Dense(10, activation='softmax', name='output')
])

print("CNN Architecture:")
print("-" * 60)
model.summary()

print(f"\nLayer Breakdown:")
print(f"  Conv1: 32 filters × (3×3) = learns 32 different patterns")
print(f"  Pool1: Reduces size by 50% (2×2 max pooling)")
print(f"  Conv2: 64 filters = learns more complex patterns")
print(f"  Pool2: Further size reduction")
print(f"  Conv3: 64 filters = even more complex patterns")
print(f"  Flatten: Convert to 1D for dense layers")
print(f"  Dense1: 128 neurons")
print(f"  Dropout: Prevent overfitting (50% dropout)")
print(f"  Output: 10 neurons (one per digit)")

print(f"\nTotal parameters: {model.count_params():,}")

# ============================================
# COMPILE MODEL
# ============================================

print("\n" + "=" * 60)
print("5. COMPILING MODEL")
print("=" * 60)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For one-hot encoded labels
    metrics=['accuracy']
)

print("✅ Model compiled:")
print("   Optimizer: Adam")
print("   Loss: Categorical Crossentropy (multi-class)")
print("   Metrics: Accuracy")

# ============================================
# TRAIN MODEL
# ============================================

print("\n" + "=" * 60)
print("6. TRAINING CNN")
print("=" * 60)

print("Training for 10 epochs...")
print("(This will take a few minutes...)\n")

history = model.fit(
    X_train, y_train_onehot,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

print("\n✅ Training complete!")

# ============================================
# EVALUATE MODEL
# ============================================

print("\n" + "=" * 60)
print("7. EVALUATING MODEL")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)

print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(f"\nConfusion Matrix:")
print(cm)

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Find misclassified examples
misclassified = np.where(y_pred != y_test)[0]
print(f"\nMisclassified: {len(misclassified)} out of {len(y_test)}")
print(f"Error rate: {len(misclassified)/len(y_test):.2%}")

# ============================================
# VISUALIZE FILTERS
# ============================================

print("\n" + "=" * 60)
print("8. VISUALIZING LEARNED FILTERS")
print("=" * 60)

# Get first convolutional layer
conv1_layer = model.get_layer('conv1')
filters, biases = conv1_layer.get_weights()

print(f"First Conv Layer Filters:")
print(f"  Shape: {filters.shape}")
print(f"  32 filters of size 3×3×1")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("9. CREATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

fig.suptitle('CNN for MNIST Digit Classification', fontsize=20, fontweight='bold')

# Plot 1: Sample Images
ax1 = fig.add_subplot(gs[0, :2])
ax1.axis('off')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f'Label: {y_train[i]}', fontsize=10)
    plt.axis('off')
plt.suptitle('Sample Training Images', x=0.25, y=0.95, fontsize=12, fontweight='bold')

# Plot 2: Training History - Accuracy
ax2 = fig.add_subplot(gs[0, 2:])
ax2.plot(history.history['accuracy'], linewidth=2.5, label='Training', color='blue')
ax2.plot(history.history['val_accuracy'], linewidth=2.5, label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Model Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Training History - Loss
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(history.history['loss'], linewidth=2.5, label='Training', color='blue')
ax3.plot(history.history['val_loss'], linewidth=2.5, label='Validation', color='red')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax3.set_title('Model Loss', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = fig.add_subplot(gs[1, 2:])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar_kws={'shrink': 0.8})
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# Plot 5: First Layer Filters
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
for i in range(32):
    plt.subplot(4, 8, i + 1)
    f = filters[:, :, 0, i]
    plt.imshow(f, cmap='gray')
    plt.axis('off')
plt.suptitle('32 Learned Filters (First Conv Layer)', x=0.5, y=0.68, 
            fontsize=12, fontweight='bold')

# Plot 6: Correct Predictions
ax6 = fig.add_subplot(gs[3, :2])
ax6.axis('off')
correct = np.where(y_pred == y_test)[0][:10]
for i, idx in enumerate(correct):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}', fontsize=9, color='green')
    plt.axis('off')
plt.suptitle('Correct Predictions ✓', x=0.25, y=0.31, fontsize=12, 
            fontweight='bold', color='green')

# Plot 7: Misclassified Examples
ax7 = fig.add_subplot(gs[3, 2:])
ax7.axis('off')
if len(misclassified) >= 10:
    sample_mis = misclassified[:10]
else:
    sample_mis = misclassified
    
for i, idx in enumerate(sample_mis):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}', fontsize=9, color='red')
    plt.axis('off')
plt.suptitle('Misclassified Examples ✗', x=0.75, y=0.31, fontsize=12,
            fontweight='bold', color='red')

plt.savefig('plots/48_cnn_mnist.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/48_cnn_mnist.png")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("10. SAVING MODEL")
print("=" * 60)

model.save('models/mnist_cnn.keras')
print("✅ Model saved: models/mnist_cnn.keras")

# Test prediction
print("\nTesting saved model...")
loaded_model = keras.models.load_model('models/mnist_cnn.keras')
sample_pred = loaded_model.predict(X_test[0:1], verbose=0)
predicted_digit = np.argmax(sample_pred)
actual_digit = y_test[0]

print(f"✅ Loaded model works!")
print(f"   Actual digit: {actual_digit}")
print(f"   Predicted digit: {predicted_digit}")
print(f"   Confidence: {sample_pred[0][predicted_digit]:.2%}")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: CNNs")
print("=" * 60)

print(f"""
WHAT WE LEARNED:

1. CNN ARCHITECTURE:
   • Convolutional layers: Learn spatial patterns
   • Pooling layers: Reduce dimensions
   • Dense layers: Final classification
   • Total parameters: {model.count_params():,}

2. WHY CNNs WORK:
   • Shared weights: Same filter across image
   • Local connectivity: Neurons see small regions
   • Translation invariance: Detect patterns anywhere
   • Hierarchical learning: Simple → Complex features

3. LAYER TYPES:
   • Conv2D: Learns filters/patterns
   • MaxPooling2D: Downsamples (takes max)
   • Dropout: Prevents overfitting
   • Dense: Traditional neural network
   • Flatten: Converts 2D → 1D

4. RESULTS:
   • Test Accuracy: {test_acc:.2%}
   • Misclassified: {len(misclassified)}/{len(y_test)}
   • Training time: ~5-10 minutes (CPU)

5. ACTIVATION FUNCTIONS:
   • ReLU: Hidden layers (fast, effective)
   • Softmax: Output (probability distribution)

6. DATA REQUIREMENTS:
   • Normalized: 0-1 range
   • Reshaped: (samples, height, width, channels)
   • One-hot: Multi-class labels

7. LEARNED PATTERNS:
   • Layer 1: Edges, lines
   • Layer 2: Shapes, curves
   • Layer 3: Complex patterns
   • Dense: Combines for classification

REAL-WORLD APPLICATIONS:
  • Facial recognition
  • Self-driving cars (lane detection)
  • Medical imaging (tumor detection)
  • Object detection (YOLO, R-CNN)
  • Image generation (GANs)

NEXT: Color images with CIFAR-10!
""")

print("\n✅ CNN fundamentals complete!")