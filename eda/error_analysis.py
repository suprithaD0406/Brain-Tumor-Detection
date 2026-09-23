import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.efficientnet import preprocess_input# ==========================================================
# SETTINGS
# ==========================================================

DATASET_PATH = Path("dataset")

MODEL_PATH = "saved_models/final_brain_tumor_model.keras"

IMAGE_SIZE = (224, 224)

RANDOM_STATE = 42

BATCH_SIZE = 32# ==========================================================
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

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

class_names = encoder.classes_

NUM_CLASSES = len(class_names)# ==========================================================
# CREATE TEST SPLIT
# ==========================================================

_, test_paths, _, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels,
)# ==========================================================
# IMAGE LOADER
# ==========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = preprocess_input(image)

    return image, label# ==========================================================
# CREATE TEST DATASET
# ==========================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Model...\n")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model Loaded Successfully!\n")# ==========================================================
# PREDICT
# ==========================================================

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
    [label.numpy() for _, label in test_dataset]
)# ==========================================================
# FIND WRONG PREDICTIONS
# ==========================================================

wrong_predictions = []

for i in range(len(test_paths)):

    actual = class_names[
        true_labels[i]
    ]

    predicted = class_names[
        predicted_labels[i]
    ]

    confidence = float(
        np.max(predictions[i]) * 100
    )

    if actual != predicted:

        wrong_predictions.append(

            {

                "Image Path": test_paths[i],
                "Actual": actual,
                "Predicted": predicted,
                "Confidence (%)": round(confidence, 2),

            }

        )# ==========================================================
# SAVE REPORT
# ==========================================================

os.makedirs(
    "outputs",
    exist_ok=True,
)

df = pd.DataFrame(
    wrong_predictions
)

df.to_csv(
    "outputs/wrong_predictions.csv",
    index=False,
)

print("\n===================================")
print(f"Total Wrong Predictions : {len(df)}")
print("Report Saved Successfully!")
print("Location : outputs/wrong_predictions.csv")
print("===================================\n")