"""
Day 13: Advanced Transfer Learning
Comparing multiple pre-trained models and fine-tuning strategies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ADVANCED TRANSFER LEARNING")
print("=" * 60)

# ============================================
# TRANSFER LEARNING STRATEGIES
# ============================================

print("\n1. TRANSFER LEARNING STRATEGIES")
print("-" * 60)

print("""
FOUR TRANSFER LEARNING APPROACHES:

1. FEATURE EXTRACTION (Fastest):
   • Freeze ALL pre-trained layers
   • Only train new classifier on top
   • Use: Small dataset, similar to ImageNet
   • Time: Minutes

2. FINE-TUNING (Most Common):
   • Freeze most layers
   • Unfreeze last few layers
   • Train with low learning rate
   • Use: Medium dataset, somewhat similar
   • Time: Hours

3. FULL FINE-TUNING (Slower):
   • Unfreeze ALL layers
   • Train with very low learning rate
   • Use: Large dataset
   • Time: Hours to days

4. TRAIN FROM SCRATCH (Rare):
   • Don't use pre-trained weights
   • Random initialization
   • Use: Very different from ImageNet
   • Time: Days to weeks

TODAY'S APPROACH:
  Feature Extraction → Fine-tuning
  (Fast initial training, then optimize)

MODELS WE'LL COMPARE:
  • MobileNetV2: Fast, mobile-optimized
  • ResNet50: Deep, residual connections
  • EfficientNetB0: State-of-the-art efficiency
  • VGG16: Classic, simple architecture
""")

# ============================================
# LOAD DATA
# ============================================

print("\n" + "=" * 60)
print("2. LOADING CIFAR-10")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Use subset for faster experimentation
train_samples = 40000  # Use 40k instead of 50k
X_train = X_train[:train_samples]
y_train = y_train[:train_samples]

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot
y_train_onehot = keras.utils.to_categorical(y_train, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

print(f"✅ Data prepared:")
print(f"   Training: {X_train.shape[0]:,} images")
print(f"   Test: {X_test.shape[0]:,} images")
print(f"   Using subset for faster training")

# ============================================
# HELPER FUNCTION
# ============================================

def build_transfer_model(base_model, model_name):
    """Build transfer learning model with given base"""
    
    # Freeze base
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name=model_name)
    
    return model

# ============================================
# MODEL 1: MOBILENETV2
# ============================================

print("\n" + "=" * 60)
print("3. MODEL 1: MobileNetV2")
print("=" * 60)

print("Loading MobileNetV2...")
base_mobile = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

model_mobile = build_transfer_model(base_mobile, 'MobileNetV2')

print(f"\n✅ MobileNetV2 loaded:")
print(f"   Parameters: {base_mobile.count_params():,}")
print(f"   Layers: {len(base_mobile.layers)}")

model_mobile.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining MobileNetV2 (5 epochs)...")
start_time = time.time()

history_mobile = model_mobile.fit(
    X_train, y_train_onehot,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

mobile_time = time.time() - start_time

# Evaluate
mobile_loss, mobile_acc = model_mobile.evaluate(X_test, y_test_onehot, verbose=0)

print(f"\n✅ MobileNetV2 Results:")
print(f"   Test Accuracy: {mobile_acc:.2%}")
print(f"   Training Time: {mobile_time:.1f} seconds")

# ============================================
# MODEL 2: RESNET50
# ============================================

print("\n" + "=" * 60)
print("4. MODEL 2: ResNet50")
print("=" * 60)

print("Loading ResNet50...")
base_resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

model_resnet = build_transfer_model(base_resnet, 'ResNet50')

print(f"\n✅ ResNet50 loaded:")
print(f"   Parameters: {base_resnet.count_params():,}")
print(f"   Layers: {len(base_resnet.layers)}")
print(f"   Key feature: Residual connections (skip connections)")

model_resnet.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining ResNet50 (5 epochs)...")
start_time = time.time()

history_resnet = model_resnet.fit(
    X_train, y_train_onehot,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

resnet_time = time.time() - start_time

# Evaluate
resnet_loss, resnet_acc = model_resnet.evaluate(X_test, y_test_onehot, verbose=0)

print(f"\n✅ ResNet50 Results:")
print(f"   Test Accuracy: {resnet_acc:.2%}")
print(f"   Training Time: {resnet_time:.1f} seconds")

# ============================================
# MODEL 3: EFFICIENTNETB0
# ============================================

print("\n" + "=" * 60)
print("5. MODEL 3: EfficientNetB0")
print("=" * 60)

print("Loading EfficientNetB0...")

try:
    base_efficient = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )
    
    model_efficient = build_transfer_model(base_efficient, 'EfficientNetB0')
    
    print(f"\n✅ EfficientNetB0 loaded:")
    print(f"   Parameters: {base_efficient.count_params():,}")
    print(f"   Layers: {len(base_efficient.layers)}")
    print(f"   Key feature: Compound scaling (width, depth, resolution)")
    
    model_efficient.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining EfficientNetB0 (5 epochs)...")
    start_time = time.time()
    
    history_efficient = model_efficient.fit(
        X_train, y_train_onehot,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    efficient_time = time.time() - start_time
    
    # Evaluate
    efficient_loss, efficient_acc = model_efficient.evaluate(X_test, y_test_onehot, verbose=0)
    
    print(f"\n✅ EfficientNetB0 Results:")
    print(f"   Test Accuracy: {efficient_acc:.2%}")
    print(f"   Training Time: {efficient_time:.1f} seconds")
    
    efficient_available = True

except Exception as e:
    print(f"⚠️  EfficientNet not available: {e}")
    print("   Skipping EfficientNet (may need newer TensorFlow)")
    efficient_available = False
    efficient_acc = 0
    efficient_time = 0

# ============================================
# FINE-TUNING BEST MODEL
# ============================================

print("\n" + "=" * 60)
print("6. FINE-TUNING BEST MODEL")
print("=" * 60)

# Find best model
results = {
    'MobileNetV2': mobile_acc,
    'ResNet50': resnet_acc,
}
if efficient_available:
    results['EfficientNetB0'] = efficient_acc

best_model_name = max(results, key=results.get)
best_acc = results[best_model_name]

print(f"Best model: {best_model_name} ({best_acc:.2%})")

# Fine-tune the best model
if best_model_name == 'MobileNetV2':
    best_model = model_mobile
    best_base = base_mobile
elif best_model_name == 'ResNet50':
    best_model = model_resnet
    best_base = base_resnet
else:
    best_model = model_efficient
    best_base = base_efficient

print(f"\nFine-tuning {best_model_name}...")
print("Strategy: Unfreeze last 20 layers, train with low LR")

# Unfreeze last layers
best_base.trainable = True

# Freeze all except last 20
for layer in best_base.layers[:-20]:
    layer.trainable = False

trainable = sum([np.prod(v.shape) for v in best_model.trainable_weights])
non_trainable = sum([np.prod(v.shape) for v in best_model.non_trainable_weights])

print(f"\nTrainable parameters: {trainable:,}")
print(f"Non-trainable parameters: {non_trainable:,}")

# Re-compile with lower learning rate
best_model.compile(
    optimizer=keras.optimizers.Adam(0.0001),  # 10x lower!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nFine-tuning for 5 epochs...")
history_finetune = best_model.fit(
    X_train, y_train_onehot,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Final evaluation
final_loss, final_acc = best_model.evaluate(X_test, y_test_onehot, verbose=0)

print(f"\n✅ Fine-tuning complete!")
print(f"   Before: {best_acc:.2%}")
print(f"   After: {final_acc:.2%}")
print(f"   Improvement: +{(final_acc - best_acc)*100:.2f}%")

# ============================================
# COMPARISON
# ============================================

print("\n" + "=" * 60)
print("7. MODEL COMPARISON")
print("=" * 60)

comparison_data = {
    'Model': ['MobileNetV2', 'ResNet50'],
    'Parameters': [
        f"{base_mobile.count_params()/1e6:.1f}M",
        f"{base_resnet.count_params()/1e6:.1f}M"
    ],
    'Accuracy': [mobile_acc, resnet_acc],
    'Training Time': [f"{mobile_time:.0f}s", f"{resnet_time:.0f}s"]
}

if efficient_available:
    comparison_data['Model'].append('EfficientNetB0')
    comparison_data['Parameters'].append(f"{base_efficient.count_params()/1e6:.1f}M")
    comparison_data['Accuracy'].append(efficient_acc)
    comparison_data['Training Time'].append(f"{efficient_time:.0f}s")

comparison_data['Model'].append(f'{best_model_name} (Fine-tuned)')
comparison_data['Parameters'].append('Same')
comparison_data['Accuracy'].append(final_acc)
comparison_data['Training Time'].append('5 more epochs')

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================
# DETAILED EVALUATION
# ============================================

print("\n" + "=" * 60)
print("8. DETAILED EVALUATION")
print("=" * 60)

# Predictions with best model
y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Per-class accuracy
print(f"\nPer-Class Accuracy ({best_model_name}):")
for i, class_name in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {class_name:12s}: {class_acc:.2%}")

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================
# SAVE BEST MODEL
# ============================================

print("\n" + "=" * 60)
print("9. SAVING BEST MODEL")
print("=" * 60)

best_model.save(f'models/{best_model_name.lower()}_finetuned.keras')
print(f"✅ Saved: models/{best_model_name.lower()}_finetuned.keras")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("10. CREATING VISUALIZATIONS")
print("=" * 60)

import pandas as pd

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

fig.suptitle('Transfer Learning Model Comparison', fontsize=20, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = fig.add_subplot(gs[0, :2])
models = comparison_data['Model']
accuracies = comparison_data['Accuracy']
colors = ['skyblue', 'lightcoral']
if efficient_available:
    colors.append('lightgreen')
colors.append('gold')

bars = ax1.bar(range(len(models)), accuracies, color=colors, 
              edgecolor='black', linewidth=2)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0.6, 0.9)
ax1.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
            f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Parameters vs Accuracy
ax2 = fig.add_subplot(gs[0, 2])
params_millions = [
    base_mobile.count_params()/1e6,
    base_resnet.count_params()/1e6
]
acc_for_scatter = [mobile_acc, resnet_acc]
model_names_scatter = ['MobileNetV2', 'ResNet50']

if efficient_available:
    params_millions.append(base_efficient.count_params()/1e6)
    acc_for_scatter.append(efficient_acc)
    model_names_scatter.append('EfficientNet')

scatter = ax2.scatter(params_millions, acc_for_scatter, s=300, 
                     alpha=0.6, edgecolors='black', linewidth=2)

for i, name in enumerate(model_names_scatter):
    ax2.annotate(name, (params_millions[i], acc_for_scatter[i]),
                fontsize=9, ha='center', va='bottom')

ax2.set_xlabel('Parameters (Millions)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Parameters vs Accuracy', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: MobileNetV2 Training History
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(history_mobile.history['accuracy'], linewidth=2.5, 
        label='Training', color='blue')
ax3.plot(history_mobile.history['val_accuracy'], linewidth=2.5,
        label='Validation', color='red')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('MobileNetV2 Training', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: ResNet50 Training History
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(history_resnet.history['accuracy'], linewidth=2.5,
        label='Training', color='green')
ax4.plot(history_resnet.history['val_accuracy'], linewidth=2.5,
        label='Validation', color='orange')
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('ResNet50 Training', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# Plot 5: Fine-tuning History
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(history_finetune.history['accuracy'], linewidth=2.5,
        label='Training', color='purple')
ax5.plot(history_finetune.history['val_accuracy'], linewidth=2.5,
        label='Validation', color='pink')
ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax5.set_title(f'{best_model_name} Fine-tuning', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# Plot 6: Confusion Matrix
ax6 = fig.add_subplot(gs[2, :])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
           xticklabels=class_names, yticklabels=class_names,
           cbar_kws={'shrink': 0.8})
ax6.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax6.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax6.set_title(f'Confusion Matrix - {best_model_name} (Fine-tuned)', 
             fontsize=14, fontweight='bold')
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right', fontsize=10)
plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=10)

plt.savefig('plots/51_transfer_learning_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/51_transfer_learning_comparison.png")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: TRANSFER LEARNING")
print("=" * 60)

print(f"""
MODELS COMPARED:
  1. MobileNetV2: {mobile_acc:.2%} ({mobile_time:.0f}s)
  2. ResNet50: {resnet_acc:.2%} ({resnet_time:.0f}s)
  {'3. EfficientNetB0: ' + f'{efficient_acc:.2%} ({efficient_time:.0f}s)' if efficient_available else ''}

BEST MODEL: {best_model_name}
  • Initial accuracy: {best_acc:.2%}
  • After fine-tuning: {final_acc:.2%}
  • Improvement: +{(final_acc - best_acc)*100:.2f}%

KEY LEARNINGS:

1. PRE-TRAINED MODELS WORK:
   • Trained on ImageNet (1.2M images)
   • Transfer knowledge to CIFAR-10
   • 5 epochs vs 50 (10x faster!)

2. FINE-TUNING HELPS:
   • Unfreeze last layers
   • Low learning rate (0.0001)
   • +{(final_acc - best_acc)*100:.1f}% improvement

3. MODEL CHOICE MATTERS:
   • MobileNetV2: Fast, mobile-friendly
   • ResNet50: Deeper, residual connections
   • Different architectures, different strengths

4. EFFICIENCY vs ACCURACY:
   • More parameters ≠ always better
   • MobileNetV2: Fewer params, fast, good accuracy
   • ResNet50: More params, slower, similar accuracy

5. WHEN TO USE EACH:
   • MobileNetV2: Mobile apps, edge devices
   • ResNet50: Server deployment, need best accuracy
   • EfficientNet: Best accuracy/efficiency ratio

REAL-WORLD STRATEGY:
  1. Start with MobileNetV2 (fast baseline)
  2. Try ResNet50 (if time permits)
  3. Fine-tune best model
  4. Deploy!

NEXT: Build custom image classifier!
""")

print("\n✅ Transfer learning comparison complete!")