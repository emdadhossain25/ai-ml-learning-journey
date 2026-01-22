"""
Day 13: Custom Image Classifier
Framework for building classifiers on any image dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CUSTOM IMAGE CLASSIFIER FRAMEWORK")
print("=" * 60)

# ============================================
# WHAT IS THIS?
# ============================================

print("\n1. BUILDING CUSTOM IMAGE CLASSIFIERS")
print("-" * 60)

print("""
CUSTOM IMAGE CLASSIFIER: Train on YOUR images

USE CASES:
  • Medical: X-ray diagnosis, skin lesion classification
  • Agriculture: Crop disease detection, weed identification
  • Manufacturing: Product defect detection, quality control
  • Wildlife: Animal species identification
  • Retail: Product categorization, visual search
  • Security: Face recognition, object detection

REQUIREMENTS:
  • Images organized in folders (one per class)
  • At least 100-500 images per class
  • Similar image sizes (will resize)
  • Labeled correctly

EXAMPLE STRUCTURE:
  dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class3/
    │       └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── class3/

TODAY'S DEMO:
  We'll create a synthetic dataset for demonstration,
  then show you how to use your own images!
""")

# ============================================
# CREATE DEMO DATASET
# ============================================

print("\n" + "=" * 60)
print("2. CREATING DEMO DATASET")
print("=" * 60)

print("Creating synthetic image dataset for demonstration...")

# Create directory structure
os.makedirs('custom_dataset/train/circles', exist_ok=True)
os.makedirs('custom_dataset/train/squares', exist_ok=True)
os.makedirs('custom_dataset/train/triangles', exist_ok=True)
os.makedirs('custom_dataset/test/circles', exist_ok=True)
os.makedirs('custom_dataset/test/squares', exist_ok=True)
os.makedirs('custom_dataset/test/triangles', exist_ok=True)

def create_circle_image(size=64):
    """Create image with circle"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 3
    
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                img[i, j] = [255, 0, 0]  # Red circle
    
    return img

def create_square_image(size=64):
    """Create image with square"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    margin = size // 4
    img[margin:-margin, margin:-margin] = [0, 255, 0]  # Green square
    
    return img

def create_triangle_image(size=64):
    """Create image with triangle"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            # Simple triangle condition
            if j > size//4 and j < 3*size//4 and i > size//4 and i < j + size//4 and i < size - j + 3*size//4:
                img[i, j] = [0, 0, 255]  # Blue triangle
    
    return img

# Generate training images
print("Generating training images...")
np.random.seed(42)

for i in range(200):
    # Circles
    img = create_circle_image()
    # Add noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'custom_dataset/train/circles/circle_{i}.png')
    
    # Squares
    img = create_square_image()
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'custom_dataset/train/squares/square_{i}.png')
    
    # Triangles
    img = create_triangle_image()
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'custom_dataset/train/triangles/triangle_{i}.png')

# Generate test images
print("Generating test images...")
for i in range(50):
    # Circles
    img = create_circle_image()
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'custom_dataset/test/circles/circle_{i}.png')
    
    # Squares
    img = create_square_image()
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'custom_dataset/test/squares/square_{i}.png')
    
    # Triangles
    img = create_triangle_image()
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'custom_dataset/test/triangles/triangle_{i}.png')

print(f"\n✅ Demo dataset created:")
print(f"   Training images: 600 (200 per class)")
print(f"   Test images: 150 (50 per class)")
print(f"   Classes: circles, squares, triangles")
print(f"   Location: custom_dataset/")

# ============================================
# DATA GENERATORS
# ============================================

print("\n" + "=" * 60)
print("3. CREATING DATA GENERATORS")
print("=" * 60)

print("""
DATA GENERATORS: Load images on-the-fly
  • Memory efficient (don't load all at once)
  • Automatic augmentation
  • Automatic batching
  • Automatic resizing
""")

# Image size
IMG_SIZE = 64
BATCH_SIZE = 32

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 20% for validation
)

# Test data generator (no augmentation!)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    'custom_dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'custom_dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'custom_dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n✅ Data generators created:")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {validation_generator.samples}")
print(f"   Test samples: {test_generator.samples}")
print(f"   Classes: {train_generator.class_indices}")
print(f"   Batch size: {BATCH_SIZE}")

# ============================================
# BUILD MODEL - OPTION 1: FROM SCRATCH
# ============================================

print("\n" + "=" * 60)
print("4. OPTION 1: TRAIN FROM SCRATCH")
print("=" * 60)

model_scratch = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes
], name='FromScratch')

print("Model from scratch:")
print("-" * 60)
model_scratch.summary()

model_scratch.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining from scratch (10 epochs)...")

history_scratch = model_scratch.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=1
)

# Evaluate
scratch_loss, scratch_acc = model_scratch.evaluate(test_generator, verbose=0)

print(f"\n✅ From Scratch Results:")
print(f"   Test Accuracy: {scratch_acc:.2%}")

# ============================================
# BUILD MODEL - OPTION 2: TRANSFER LEARNING
# ============================================

print("\n" + "=" * 60)
print("5. OPTION 2: TRANSFER LEARNING")
print("=" * 60)

print("Loading MobileNetV2 as base model...")

# Load pre-trained base
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base
base_model.trainable = False

# Build transfer learning model
model_transfer = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
], name='TransferLearning')

print("\nTransfer learning model:")
print("-" * 60)
model_transfer.summary()

model_transfer.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining with transfer learning (10 epochs)...")

history_transfer = model_transfer.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=1
)

# Evaluate
transfer_loss, transfer_acc = model_transfer.evaluate(test_generator, verbose=0)

print(f"\n✅ Transfer Learning Results:")
print(f"   Test Accuracy: {transfer_acc:.2%}")

# ============================================
# COMPARISON
# ============================================

print("\n" + "=" * 60)
print("6. COMPARING APPROACHES")
print("=" * 60)

comparison = {
    'Approach': ['From Scratch', 'Transfer Learning'],
    'Test Accuracy': [scratch_acc, transfer_acc],
    'Training Time': ['~2 min', '~2 min'],
    'Parameters': [
        f"{model_scratch.count_params():,}",
        f"{model_transfer.count_params():,}"
    ]
}

import pandas as pd
comparison_df = pd.DataFrame(comparison)
print("\n" + comparison_df.to_string(index=False))

winner = 'Transfer Learning' if transfer_acc > scratch_acc else 'From Scratch'
print(f"\n🏆 Winner: {winner}")

# ============================================
# PREDICTIONS ON NEW IMAGES
# ============================================

print("\n" + "=" * 60)
print("7. MAKING PREDICTIONS")
print("=" * 60)

# Choose best model
best_model = model_transfer if transfer_acc > scratch_acc else model_scratch

# Get predictions
test_generator.reset()
predictions = best_model.predict(test_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

# Class names
class_names = list(test_generator.class_indices.keys())

print(f"Sample predictions:")
for i in range(10):
    true_label = class_names[true_classes[i]]
    pred_label = class_names[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]]
    
    status = "✓" if true_label == pred_label else "✗"
    print(f"  {status} True: {true_label:10s} | Pred: {pred_label:10s} ({confidence:.2%})")

# Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(true_classes, predicted_classes)
cm = confusion_matrix(true_classes, predicted_classes)

print(f"\nOverall Test Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("8. SAVING MODEL")
print("=" * 60)

best_model.save('models/custom_shape_classifier.keras')
print("✅ Model saved: models/custom_shape_classifier.keras")

# Save class names
import json
with open('models/custom_shape_classes.json', 'w') as f:
    json.dump(class_names, f)
print("✅ Class names saved: models/custom_shape_classes.json")

# ============================================
# HOW TO USE YOUR OWN IMAGES
# ============================================

print("\n" + "=" * 60)
print("9. USING YOUR OWN IMAGES")
print("=" * 60)

print("""
TO TRAIN ON YOUR OWN IMAGES:

1. ORGANIZE YOUR IMAGES:
   my_dataset/
     ├── train/
     │   ├── class1/
     │   │   ├── img1.jpg
     │   │   ├── img2.jpg
     │   │   └── ...
     │   ├── class2/
     │   └── class3/
     └── test/
         ├── class1/
         ├── class2/
         └── class3/

2. MODIFY THE CODE:
   train_generator = train_datagen.flow_from_directory(
       'my_dataset/train',  # Your path here
       target_size=(64, 64),  # Or (224, 224) for better accuracy
       batch_size=32,
       class_mode='categorical'
   )

3. ADJUST MODEL:
   # Change last layer to match your classes
   layers.Dense(num_classes, activation='softmax')

4. TRAIN:
   history = model.fit(
       train_generator,
       epochs=20,  # More epochs for real data
       validation_data=validation_generator
   )

5. EVALUATE & SAVE:
   model.save('my_classifier.keras')

TIPS FOR BEST RESULTS:
  • Use at least 100-500 images per class
  • More data = better accuracy
  • Balance classes (similar number per class)
  • Use data augmentation
  • Start with transfer learning (MobileNetV2)
  • Fine-tune if needed
  • Monitor validation accuracy (not just training)
""")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("10. CREATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)

fig.suptitle('Custom Image Classifier - Shapes Recognition', 
             fontsize=20, fontweight='bold')

# Plot 1: Sample Training Images
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

# Get sample batch
sample_batch = next(train_generator)
sample_images = sample_batch[0][:9]
sample_labels = np.argmax(sample_batch[1][:9], axis=1)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f'{class_names[sample_labels[i]]}', fontsize=10)
    plt.axis('off')
plt.suptitle('Sample Training Images (With Augmentation)', 
            x=0.5, y=0.95, fontsize=13, fontweight='bold')

# Plot 2: From Scratch - Accuracy
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history_scratch.history['accuracy'], linewidth=2.5, 
        label='Training', color='blue')
ax2.plot(history_scratch.history['val_accuracy'], linewidth=2.5,
        label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('From Scratch - Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: From Scratch - Loss
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(history_scratch.history['loss'], linewidth=2.5,
        label='Training', color='blue')
ax3.plot(history_scratch.history['val_loss'], linewidth=2.5,
        label='Validation', color='red')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax3.set_title('From Scratch - Loss', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Plot 4: Transfer Learning - Accuracy
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(history_transfer.history['accuracy'], linewidth=2.5,
        label='Training', color='green')
ax4.plot(history_transfer.history['val_accuracy'], linewidth=2.5,
        label='Validation', color='orange')
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('Transfer Learning - Accuracy', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# Plot 5: Confusion Matrix
ax5 = fig.add_subplot(gs[2, :2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=class_names, yticklabels=class_names,
           annot_kws={'fontsize': 14})
ax5.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax5.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax5.set_title(f'Confusion Matrix ({winner})', fontsize=14, fontweight='bold')

# Plot 6: Model Comparison
ax6 = fig.add_subplot(gs[2, 2])
approaches = ['From Scratch', 'Transfer Learning']
accuracies = [scratch_acc, transfer_acc]
colors_comp = ['skyblue', 'lightgreen']

bars = ax6.bar(approaches, accuracies, color=colors_comp,
              edgecolor='black', linewidth=2)
ax6.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax6.set_title('Approach Comparison', fontsize=14, fontweight='bold')
ax6.set_ylim(0.8, 1.0)
ax6.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    ax6.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
            f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 7: Sample Predictions
ax7 = fig.add_subplot(gs[3, :])
ax7.axis('off')

# Get sample test images
test_generator.reset()
test_batch = next(test_generator)
test_images = test_batch[0][:10]
test_true = np.argmax(test_batch[1][:10], axis=1)

# Predict
test_predictions = best_model.predict(test_images, verbose=0)
test_pred = np.argmax(test_predictions, axis=1)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i])
    
    true_label = class_names[test_true[i]]
    pred_label = class_names[test_pred[i]]
    confidence = test_predictions[i][test_pred[i]]
    
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'T:{true_label}\nP:{pred_label}\n{confidence:.0%}', 
             fontsize=9, color=color)
    plt.axis('off')

plt.suptitle('Test Predictions (Green=Correct, Red=Wrong)', 
            x=0.5, y=0.31, fontsize=13, fontweight='bold')

plt.savefig('plots/52_custom_image_classifier.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/52_custom_image_classifier.png")

# ============================================
# DEPLOYMENT CODE
# ============================================

print("\n" + "=" * 60)
print("11. DEPLOYMENT CODE")
print("=" * 60)

deployment_code = '''
# ============================================
# HOW TO USE SAVED MODEL FOR PREDICTIONS
# ============================================

from tensorflow import keras
import numpy as np
from PIL import Image
import json

# Load model and class names
model = keras.models.load_model('models/custom_shape_classifier.keras')
with open('models/custom_shape_classes.json', 'r') as f:
    class_names = json.load(f)

def predict_image(image_path):
    """Predict class of a single image"""
    # Load and preprocess
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return {
        'class': class_names[predicted_class],
        'confidence': float(confidence),
        'all_probabilities': {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
    }

# Example usage:
result = predict_image('new_image.png')
print(f"Predicted: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
'''

# Save deployment code
with open('models/how_to_use_classifier.py', 'w') as f:
    f.write(deployment_code)

print("✅ Deployment code saved: models/how_to_use_classifier.py")

print("""
To use your trained model:

1. Load model:
   model = keras.models.load_model('models/custom_shape_classifier.keras')

2. Prepare new image:
   img = Image.open('new_image.jpg').resize((64, 64))
   img_array = np.array(img) / 255.0
   img_array = np.expand_dims(img_array, 0)

3. Predict:
   predictions = model.predict(img_array)
   predicted_class = np.argmax(predictions)
   
4. Get result:
   print(f"Class: {class_names[predicted_class]}")
   print(f"Confidence: {predictions[0][predicted_class]:.2%}")
""")

# ============================================
# KEY TAKEAWAYS
# ============================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS: CUSTOM IMAGE CLASSIFIER")
print("=" * 60)

print(f"""
WHAT WE BUILT:
  • Shape classifier (circles, squares, triangles)
  • Two approaches: Scratch vs Transfer Learning
  • Complete training pipeline
  • Deployment-ready code

RESULTS:
  • From Scratch: {scratch_acc:.2%}
  • Transfer Learning: {transfer_acc:.2%}
  • Winner: {winner}

KEY LEARNINGS:

1. DATA ORGANIZATION:
   • Folder per class
   • Automatic loading with flow_from_directory
   • Validation split built-in

2. DATA AUGMENTATION:
   • Rotation, shifts, flips, zoom
   • Prevents overfitting
   • Simulates real-world variations

3. TWO APPROACHES:
   a) From Scratch:
      • Full control
      • Needs more data
      • Longer training
   
   b) Transfer Learning:
      • Faster training
      • Better with less data
      • Pre-trained knowledge

4. DEPLOYMENT:
   • Save model (.keras)
   • Save class names (.json)
   • Simple prediction function
   • Production-ready!

APPLY TO YOUR DATA:
  1. Organize images in folders
  2. Adjust target_size and num_classes
  3. Choose approach (transfer learning recommended)
  4. Train for 20-50 epochs
  5. Evaluate on test set
  6. Save and deploy!

REAL-WORLD USE:
  • Medical: Disease classification
  • Agriculture: Crop monitoring
  • Retail: Visual search
  • Security: Object recognition
  • Manufacturing: Defect detection

NEXT STEPS:
  • Try with your own images
  • Experiment with different architectures
  • Add more augmentation
  • Deploy as web API
  • Build mobile app

YOU NOW KNOW:
  ✓ How to train on custom images
  ✓ When to use transfer learning
  ✓ How to deploy models
  ✓ How to make predictions
  ✓ Production ML workflow
""")

print("\n✅ Custom image classifier complete!")
print("\n🎉 You can now classify ANY images! 🎉")