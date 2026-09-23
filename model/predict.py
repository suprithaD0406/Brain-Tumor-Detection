import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

# ======================================================
# SETTINGS
# ======================================================

MODEL_PATH = "saved_models/final_brain_tumor_model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

# ======================================================
# LOAD MODEL
# ======================================================

print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!\n")

# ======================================================
# IMAGE PATH
# ======================================================

image_path = Path(input("Enter Image Path : ").strip()).resolve()

if not image_path.exists():
    print("\nImage not found!")
    exit()

# ======================================================
# PREPROCESS IMAGE
# ======================================================

img = image.load_img(
    str(image_path),
    target_size=IMAGE_SIZE,
)
img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = preprocess_input(img)

# ======================================================
# PREDICTION
# ======================================================

prediction = model.predict(img, verbose=0)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction)

# ======================================================
# RESULT
# ======================================================

print("\n==============================")

print(f"Prediction : {CLASS_NAMES[predicted_class]}")

print(f"Confidence : {confidence*100:.2f}%")

print("==============================")

print("\nProbability of Each Class\n")

for i in range(len(CLASS_NAMES)):

    print(f"{CLASS_NAMES[i]:15} : {prediction[0][i]*100:.2f}%")