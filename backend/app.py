import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI(title="Brain Tumor Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = tf.keras.models.load_model(
    "saved_models/final_brain_tumor_model.keras"
)
print("\nLoaded Model Successfully")
print(model.input_shape)
print(model.output_shape)
print(model.name)
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

IMAGE_SIZE = (224, 224)

# ---------------------------------------------------
# Home
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Brain Tumor Detection API Running"
    }

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BILINEAR,
    )

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(
        image,
        axis=0,
    )

    image = preprocess_input(image)

    prediction = model.predict(
        image,
        verbose=0,
    )

    predicted_class = CLASS_NAMES[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    )

    probabilities = {}

    for i, cls in enumerate(CLASS_NAMES):

        probabilities[cls] = float(
            prediction[0][i]
        )

    return JSONResponse(

        {

            "prediction": predicted_class,

            "confidence": round(
                confidence * 100,
                2,
            ),

            "probabilities": probabilities

        }

    )