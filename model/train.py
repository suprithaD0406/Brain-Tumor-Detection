import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# ==========================================================
# SETTINGS
# ==========================================================

DATASET_PATH = Path("dataset")

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 50

RANDOM_STATE = 42

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================================
# LOAD IMAGE PATHS
# ==========================================================

image_paths = []
labels = []

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

for class_folder in DATASET_PATH.iterdir():

    if not class_folder.is_dir():
        continue

    for image in class_folder.iterdir():

        if image.suffix.lower() in VALID_EXTENSIONS:

            image_paths.append(str(image))

            labels.append(class_folder.name)

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

class_names = encoder.classes_

NUM_CLASSES = len(class_names)

print(f"Classes : {list(class_names)}")

print(f"Total Images : {len(image_paths)}")

print()

# ==========================================================
# TRAIN + TEST
# ==========================================================

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels,
)

# ==========================================================
# TRAIN + VALIDATION
# ==========================================================

train_paths, validation_paths, train_labels, validation_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train_labels,
)

print("Training Images :", len(train_paths))
print("Validation Images :", len(validation_paths))
print("Testing Images :", len(test_paths))

print("=" * 50)

# ==========================================================
# IMAGE LOADER
# ==========================================================


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize_with_pad(
    image,
    IMAGE_SIZE[0],
    IMAGE_SIZE[1],
   )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = preprocess_input(image)

    label = tf.one_hot(
        label,
        NUM_CLASSES,
    )

    return image, label
# ==========================================================
# CREATE tf.data DATASETS
# ==========================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE,
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE,
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE,
)

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.10),
    ]
)

train_dataset = train_dataset.map(
    lambda image, label: (
        data_augmentation(image, training=True),
        label,
    ),
    num_parallel_calls=AUTOTUNE,
)

train_dataset = (
    train_dataset
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
# ==========================================================
# LOAD EFFICIENTNETB0
# ==========================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ==========================================================
# BUILD MODEL
# ==========================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False,
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.30)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu",
)(x)

x = tf.keras.layers.Dropout(0.20)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
)

model.summary()
# ==========================================================
# COMPILE MODEL
# ==========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5,
    ),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint = ModelCheckpoint(
    filepath="saved_models/brain_tumor_model_v2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
]

# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    "saved_models/final_brain_tumor_model.keras"
)

print("\nTraining Completed Successfully!")

# ==========================================================
# SAVE CLASS NAMES
# ==========================================================

np.save(
    "outputs/class_names.npy",
    class_names,
)

# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

import pickle

with open(
    "outputs/history.pkl",
    "wb",
) as file:

    pickle.dump(
        history.history,
        file,
    )

print("Training History Saved")
# ==========================================================
# EVALUATE MODEL
# ==========================================================

print("\nEvaluating Model on Test Dataset...\n")

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    test_dataset,
    verbose=1,
)

print(f"\nTest Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_accuracy*100:.2f}%")
print(f"Test Precision : {test_precision:.4f}")
print(f"Test Recall    : {test_recall:.4f}")

# ==========================================================
# LOSS GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/loss_plot.png",
    dpi=300,
)

plt.close()

# ==========================================================
# SAVE TRAINING LOG
# ==========================================================

with open(
    "outputs/training_log.txt",
    "w",
) as file:

    file.write("Brain Tumor Detection Model\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"Total Classes : {NUM_CLASSES}\n")
    file.write(f"Classes       : {list(class_names)}\n\n")

    file.write(f"Training Images   : {len(train_paths)}\n")
    file.write(f"Validation Images : {len(validation_paths)}\n")
    file.write(f"Testing Images    : {len(test_paths)}\n\n")

    file.write(f"Epochs : {len(history.history['accuracy'])}\n\n")

    file.write(
        f"Final Training Accuracy   : {history.history['accuracy'][-1]:.4f}\n"
    )

    file.write(
        f"Final Validation Accuracy : {history.history['val_accuracy'][-1]:.4f}\n"
    )

    file.write(
        f"Final Training Loss       : {history.history['loss'][-1]:.4f}\n"
    )

    file.write(
        f"Final Validation Loss     : {history.history['val_loss'][-1]:.4f}\n"
    )

    file.write(f"\nTest Accuracy : {test_accuracy:.4f}\n")
    file.write(f"Test Loss     : {test_loss:.4f}\n")

print("\nTraining log saved.")

print("\nGraphs saved inside outputs/")

print("\nSaved Model:")
print("saved_models/final_brain_tumor_model.keras")

print("\nProject Training Completed Successfully!")