"""
Food Item Classifier - Model Training Script
Trains a CNN model using TensorFlow/Keras on the dataset folder.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── Configuration ────────────────────────────────────────────────────────────

# Resolve dataset path relative to this script so runs from any CWD work
DATASET_DIR   = str(Path(__file__).resolve().parent / "dataset")
MODEL_PATH    = "model/food_model.keras"
# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-4
 
# ─── Data Augmentation & Loading ──────────────────────────────────────────────

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"\nClasses ({NUM_CLASSES}): {CLASS_NAMES}\n")

# ─── Model Architecture (Transfer Learning: MobileNetV2) ──────────────────────

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze base initially

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ─── Phase 1: Train top layers ─────────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

print("\n── Phase 1: Training top layers ──")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
)

# ─── Phase 2: Fine-tune last 30 layers of base ────────────────────────────────

print("\n── Phase 2: Fine-tuning ──")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
)

# ─── Evaluate & Save ──────────────────────────────────────────────────────────

val_loss, val_acc = model.evaluate(val_generator)
print(f"\nFinal Validation Accuracy : {val_acc * 100:.2f}%")
print(f"Final Validation Loss     : {val_loss:.4f}")
print(f"\nModel saved to: {MODEL_PATH}")
print(f"Class mapping  : {train_generator.class_indices}")
