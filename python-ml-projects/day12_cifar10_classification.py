"""
Day 12: CIFAR-10 Color Image Classification
Real-world image classification with CNNs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CIFAR-10 IMAGE CLASSIFICATION")
print("=" * 60)

# ============================================
# WHAT IS CIFAR-10?
# ============================================

print("\n1. UNDERSTANDING CIFAR-10")
print("-" * 60)

print("""
CIFAR-10 DATASET:
  • 60,000 color images (32×32 pixels)
  • 10 classes: airplane, automobile, bird, cat, deer,
                dog, frog, horse, ship, truck
  • 50,000 training images
  • 10,000 test images
  • Real-world photos (not drawings)

COMPARISON TO MNIST:
┌─────────────────────────────────────────────────┐
│ MNIST              │  CIFAR-10                  │
├─────────────────────────────────────────────────┤
│ Grayscale (1 ch)   │  Color (3 channels: RGB)   │
│ 28×28 pixels       │  32×32 pixels              │
│ Digits (simple)    │  Objects (complex)         │
│ 98-99% accuracy    │  85-95% accuracy           │
│ Easy benchmark     │  Real challenge            │
└─────────────────────────────────────────────────┘

CHALLENGES:
  • Small images (32×32 - low resolution)
  • Similar classes (dog vs cat, truck vs automobile)
  • Background clutter
  • Varying lighting, angles, occlusion

STATE-OF-THE-ART:
  • ResNet: ~95% accuracy
  • Vision Transformers: ~99% accuracy
  • Our goal: 75-80% (good for basic CNN!)
""")

# ============================================
# LOAD CIFAR-10
# ============================================

print("\n" + "=" * 60)
print("2. LOADING CIFAR-10 DATASET")
print("=" * 60)

print("Downloading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\n✅ CIFAR-10 loaded:")
print(f"   Training images: {X_train.shape[0]:,}")
print(f"   Test images: {X_test.shape[0]:,}")
print(f"   Image size: {X_train.shape[1]}×{X_train.shape[2]}×{X_train.shape[3]}")
print(f"   Classes: {len(class_names)}")
print(f"   Class names: {', '.join(class_names)}")

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nClass distribution (training):")
for cls, count in zip(unique, counts):
    # print(f"   {class_names[cls[0]]}: {count}")
    print(f"   {class_names[cls]}: {count}")
# ============================================
# PREPROCESS DATA
# ============================================

print("\n" + "=" * 60)
print("3. PREPROCESSING")
print("=" * 60)

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot
y_train_onehot = keras.utils.to_categorical(y_train, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

print(f"✅ Preprocessing complete:")
print(f"   Training shape: {X_train.shape}")
print(f"   Test shape: {X_test.shape}")
print(f"   Pixel range: {X_train.min():.1f} to {X_train.max():.1f}")
print(f"   Labels: One-hot encoded")

# ============================================
# DATA AUGMENTATION
# ============================================

print("\n" + "=" * 60)
print("4. DATA AUGMENTATION")
print("=" * 60)

print("""
DATA AUGMENTATION: Create variations of training images
  • Helps model generalize better
  • Prevents overfitting
  • Simulates real-world variations

AUGMENTATION TECHNIQUES:
  • Rotation: Rotate images slightly
  • Width/Height Shift: Move image around
  • Horizontal Flip: Mirror image
  • Zoom: Slight zoom in/out

IMPORTANT: Only augment TRAINING data, not test!
""")

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(X_train)

print("✅ Data augmentation configured:")
print("   Rotation: ±15 degrees")
print("   Shift: ±10%")
print("   Horizontal flip: Yes")
print("   Zoom: ±10%")

# ============================================
# BUILD DEEPER CNN
# ============================================

print("\n" + "=" * 60)
print("5. BUILDING DEEPER CNN")
print("=" * 60)

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                 input_shape=(32, 32, 3), name='conv1_1'),
    layers.Conv2D(32, (3, 3), activation='relu', name='conv1_2'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Dropout(0.25, name='dropout1'),
    
    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2_2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Dropout(0.25, name='dropout2'),
    
    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1'),
    layers.Conv2D(128, (3, 3), activation='relu', name='conv3_2'),
    layers.MaxPooling2D((2, 2), name='pool3'),
    layers.Dropout(0.25, name='dropout3'),
    
    # Dense layers
    layers.Flatten(name='flatten'),
    layers.Dense(512, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout4'),
    layers.Dense(10, activation='softmax', name='output')
])

print("CIFAR-10 CNN Architecture:")
print("-" * 60)
model.summary()

print(f"\nArchitecture highlights:")
print(f"  • 3 Convolutional Blocks (32 → 64 → 128 filters)")
print(f"  • Each block: 2 Conv layers + MaxPooling + Dropout")
print(f"  • Padding='same': Preserve spatial dimensions")
print(f"  • Progressive filter increase: 32 → 64 → 128")
print(f"  • Dropout: 25% (conv) and 50% (dense)")
print(f"  • Total parameters: {model.count_params():,}")

# ============================================
# COMPILE WITH LEARNING RATE SCHEDULE
# ============================================

print("\n" + "=" * 60)
print("6. COMPILING WITH OPTIMIZATIONS")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled:")
print("   Optimizer: Adam (lr=0.001)")
print("   Loss: Categorical Crossentropy")
print("   Metrics: Accuracy")

# ============================================
# CALLBACKS
# ============================================

print("\n" + "=" * 60)
print("7. SETTING UP CALLBACKS")
print("=" * 60)

# Learning rate reduction
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Early stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint
checkpoint = callbacks.ModelCheckpoint(
    'models/cifar10_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("✅ Callbacks configured:")
print("   • ReduceLROnPlateau: Halve LR if stuck")
print("   • EarlyStopping: Stop if no improvement (10 epochs)")
print("   • ModelCheckpoint: Save best model")

# ============================================
# TRAIN MODEL
# ============================================

print("\n" + "=" * 60)
print("8. TRAINING CNN")
print("=" * 60)

print("Training for up to 50 epochs with data augmentation...")
print("(This will take 10-20 minutes depending on your hardware...)")
print("⏰ Perfect time for a coffee break! ☕\n")

history = model.fit(
    datagen.flow(X_train, y_train_onehot, batch_size=64),
    epochs=50,
    steps_per_epoch=len(X_train) // 64,
    validation_data=(X_test, y_test_onehot),
    callbacks=[reduce_lr, early_stop, checkpoint],
    verbose=1
)

print("\n✅ Training complete!")

# ============================================
# EVALUATE MODEL
# ============================================

print("\n" + "=" * 60)
print("9. EVALUATING MODEL")
print("=" * 60)

# Load best model
best_model = keras.models.load_model('models/cifar10_best.keras')

# Evaluate
test_loss, test_acc = best_model.evaluate(X_test, y_test_onehot, verbose=0)

print(f"Best Model Performance:")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
print(f"  Test Loss: {test_loss:.4f}")

# Predictions
y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(f"\nConfusion Matrix:")
print(cm)

# Per-class accuracy
print(f"\nPer-Class Accuracy:")
for i, class_name in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"  {class_name:12s}: {class_acc:.2%}")

# Classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================
# ANALYZE MISTAKES
# ============================================

print("\n" + "=" * 60)
print("10. ANALYZING MISTAKES")
print("=" * 60)

# Find misclassified
misclassified_idx = np.where(y_pred != y_test.flatten())[0]
print(f"Misclassified: {len(misclassified_idx)} out of {len(y_test)}")
print(f"Error rate: {len(misclassified_idx)/len(y_test):.2%}")

# Most confused pairs
confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((class_names[i], class_names[j], cm[i, j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\nTop 5 Most Confused Pairs:")
for true_class, pred_class, count in confusion_pairs[:5]:
    print(f"  {true_class} → {pred_class}: {count} times")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("11. CREATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)

fig.suptitle('CIFAR-10 Image Classification with CNN', fontsize=20, fontweight='bold')

# Plot 1: Sample Images
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(f'{class_names[y_train[i][0]]}', fontsize=10)
    plt.axis('off')
plt.suptitle('Sample Training Images', x=0.5, y=0.95, fontsize=13, fontweight='bold')

# Plot 2: Training History - Accuracy
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history.history['accuracy'], linewidth=2.5, label='Training', color='blue')
ax2.plot(history.history['val_accuracy'], linewidth=2.5, label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Model Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Training History - Loss
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(history.history['loss'], linewidth=2.5, label='Training', color='blue')
ax3.plot(history.history['val_loss'], linewidth=2.5, label='Validation', color='red')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax3.set_title('Model Loss', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Learning Rate Schedule
ax4 = fig.add_subplot(gs[1, 2])
if 'lr' in history.history:
    ax4.plot(history.history['lr'], linewidth=2.5, color='green')
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Learning rate not tracked', ha='center', va='center')
    ax4.axis('off')

# Plot 5: Confusion Matrix
ax5 = fig.add_subplot(gs[2, :])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=class_names, yticklabels=class_names,
           cbar_kws={'shrink': 0.8})
ax5.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax5.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax5.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax5.get_yticklabels(), rotation=0)

# Plot 6: Correct Predictions
ax6 = fig.add_subplot(gs[3, :2])
ax6.axis('off')
correct_idx = np.where(y_pred == y_test.flatten())[0]
sample_correct = np.random.choice(correct_idx, 10, replace=False)

for i, idx in enumerate(sample_correct):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx])
    true_label = class_names[y_test[idx][0]]
    confidence = y_pred_proba[idx][y_pred[idx]]
    plt.title(f'{true_label}\n{confidence:.1%}', fontsize=9, color='green')
    plt.axis('off')
plt.suptitle('Correct Predictions ✓', x=0.25, y=0.31, fontsize=12,
            fontweight='bold', color='green')

# Plot 7: Misclassified Examples
ax7 = fig.add_subplot(gs[3, 2])
ax7.axis('off')
if len(misclassified_idx) >= 5:
    sample_mis = np.random.choice(misclassified_idx, 5, replace=False)
    
    for i, idx in enumerate(sample_mis):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[idx])
        true_label = class_names[y_test[idx][0]]
        pred_label = class_names[y_pred[idx]]
        plt.title(f'T:{true_label}\nP:{pred_label}', fontsize=8, color='red')
        plt.axis('off')
    plt.suptitle('Misclassified ✗', x=0.75, y=0.31, fontsize=12,
                fontweight='bold', color='red')

plt.savefig('plots/49_cifar10_classification.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/49_cifar10_classification.png")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: CIFAR-10 CLASSIFICATION")
print("=" * 60)

print(f"""
WHAT WE LEARNED:

1. DEEPER ARCHITECTURE:
   • 3 blocks of (Conv → Conv → Pool → Dropout)
   • Progressive filters: 32 → 64 → 128
   • Total parameters: {model.count_params():,}
   • Deeper than MNIST model

2. DATA AUGMENTATION:
   • Rotation, shifts, flips, zoom
   • Creates variations of training data
   • Improves generalization
   • Only applied to training (not test!)

3. ADVANCED TECHNIQUES:
   • ReduceLROnPlateau: Adaptive learning rate
   • EarlyStopping: Prevents overfitting
   • ModelCheckpoint: Saves best model
   • Dropout: 25% (conv) + 50% (dense)

4. RESULTS:
   • Test Accuracy: {test_acc:.2%}
   • Training epochs: {len(history.history['loss'])}
   • Misclassified: {len(misclassified_idx)}/{len(y_test)}

5. CHALLENGES:
   • Color images: 3× more complex than grayscale
   • Low resolution: 32×32 pixels
   • Similar classes: cat/dog, truck/automobile
   • Background clutter

6. MOST CONFUSED PAIRS:
   Top confusion: {confusion_pairs[0][0]} ↔ {confusion_pairs[0][1]}

7. IMPROVEMENTS POSSIBLE:
   • Deeper networks (ResNet, VGG)
   • Transfer learning (pre-trained models)
   • More augmentation
   • Ensemble methods

COMPARISON:
  MNIST (grayscale, digits): 98-99% accuracy
  CIFAR-10 (color, objects): {test_acc:.0%} accuracy
  → Color images are MUCH harder!

REAL-WORLD APPLICATIONS:
  • Self-driving cars (vehicle detection)
  • Surveillance (person/animal detection)
  • Medical imaging (disease classification)
  • Quality control (defect detection)
  • Wildlife monitoring (species classification)

NEXT: Transfer Learning - Use pre-trained models!
""")

print("\n✅ CIFAR-10 classification complete!")