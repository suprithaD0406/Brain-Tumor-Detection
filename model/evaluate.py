import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras.applications.efficientnet import preprocess_input

# =====================================================
# SETTINGS
# =====================================================

DATASET_PATH = Path("dataset")

MODEL_PATH = "saved_models/brain_tumor_model_v2.keras"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

RANDOM_STATE = 42

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

image_paths = []
labels = []

for folder in DATASET_PATH.iterdir():

    if folder.is_dir():

        for image in folder.glob("*"):

            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:

                image_paths.append(str(image))
                labels.append(folder.name)

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

class_names = encoder.classes_

NUM_CLASSES = len(class_names)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

_, test_paths, _, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels,
)

# =====================================================
# IMAGE FUNCTION
# =====================================================

def process_image(path, label):

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

    return image, label


# =====================================================
# CREATE TEST DATASET
# =====================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_dataset = test_dataset.map(
    process_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Model...\n")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!\n")

# =====================================================
# PREDICT
# =====================================================

print("Predicting...\n")

predictions = model.predict(
    test_dataset,
    verbose=1,
)

predicted_labels = np.argmax(
    predictions,
    axis=1,
)

true_labels = np.concatenate(
    [y.numpy() for _, y in test_dataset]
)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
)

print("\n==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("==============================")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
)

print(report)

with open(
    "outputs/classification_report.txt",
    "w",
) as file:

    file.write(report)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
)

plt.close()

# =====================================================
# SAVE METRICS
# =====================================================

with open(
    "outputs/metrics.txt",
    "w",
) as file:

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")

print("\nSaved:")
print("classification_report.txt")
print("confusion_matrix.png")
print("metrics.txt")

print("\nEvaluation Completed Successfully!")