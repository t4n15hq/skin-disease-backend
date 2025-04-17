from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import keras
import os
import gdown
keras.config.enable_unsafe_deserialization()

from keras.saving import register_keras_serializable

# --- CUSTOM FUNCTION ---
@register_keras_serializable(package="Custom", name="custom_max_pool")
def custom_max_pool(x):
    return tf.reduce_max(x, axis=[1, 2], keepdims=True)

# --- CONFIG ---
IMG_SIZE = 224
NUM_CLASSES = 23

MODEL_PATHS = {
    "EfficientNetB3": "saved_models/EfficientNetB3_recovered.keras",
    "ResNet50": "saved_models/ResNet50_recovered.keras",
    "MobileNetV2": "saved_models/MobileNetV2_recovered.keras",
    "DenseNet121": "saved_models/DenseNet121_recovered.keras"
}

MODEL_DRIVE_IDS = {
    "EfficientNetB3": "14Wl06h312YuMBXmSIQF97y8dZv3zIzq8",
    "ResNet50": "10FwdcjZwgMmOHmh1a9h1cE_C-rVn5lwb",
    "MobileNetV2": "1gYdYkWMiDauWvTMPyCUyKWC6i21FQp7a",
    "DenseNet121": "1XupPeRPIomj-4KR8ri-zRte90O3QXTHg"
}

MODEL_WEIGHTS = {
    "EfficientNetB3": 0.260,
    "ResNet50": 0.256,
    "MobileNetV2": 0.222,
    "DenseNet121": 0.261
}

PREPROCESS_FUNCS = {
    "EfficientNetB3": tf.keras.applications.efficientnet.preprocess_input,
    "ResNet50": tf.keras.applications.resnet.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "DenseNet121": tf.keras.applications.densenet.preprocess_input
}

# --- Ensure saved_models/ exists ---
os.makedirs("saved_models", exist_ok=True)

# --- Download models if not present ---
def download_if_missing(file_id, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

# --- LOAD MODELS ---
print("Loading models...")
models = {}
for name, path in MODEL_PATHS.items():
    download_if_missing(MODEL_DRIVE_IDS[name], path)
    print(f"Loading {name}...")
    models[name] = tf.keras.models.load_model(path)
print("All models loaded.")

# --- FASTAPI SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UTILS ---
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image.convert("RGB")

# --- ENSEMBLE PREDICTION ---
def predict_ensemble(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image)
    ensemble_pred = np.zeros((NUM_CLASSES,))
    for name, model in models.items():
        preproc = PREPROCESS_FUNCS[name]
        img_proc = preproc(image_array.copy())
        img_proc = np.expand_dims(img_proc, axis=0)
        pred = model.predict(img_proc, verbose=0)[0]
        ensemble_pred += pred * MODEL_WEIGHTS[name]
    return ensemble_pred

# --- RE-RANK PREDICTIONS BASED ON METADATA ---
def adjust_with_metadata(predictions, metadata):
    adjusted = []
    for pred in predictions:
        label = pred["label"]
        score = pred["confidence"]

        try:
            age = int(metadata["age"])
            condition = metadata.get("condition", "").lower()
            skin_type = metadata.get("skin_type", "").lower()

            if "acne" in label.lower() and age > 40:
                score *= 0.6
            if "eczema" in label.lower() and skin_type == "dry":
                score *= 1.2
            if "warts" in label.lower() and age < 12:
                score *= 1.3
            if "fungal" in label.lower() and "itchy" in condition:
                score *= 1.2
        except Exception as e:
            print("Metadata adjustment error:", e)

        adjusted.append({"label": label, "confidence": score})

    adjusted = sorted(adjusted, key=lambda x: x["confidence"], reverse=True)
    return adjusted[:3]

# --- PREDICT ENDPOINT ---
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    race: str = Form(...),
    gender: str = Form(...),
    skin_color: str = Form(...),
    skin_type: str = Form(...),
    condition_description: str = Form(...)
):
    image = read_imagefile(await file.read())
    prediction = predict_ensemble(image)

    class_labels = [
        "Acne and Rosacea Photos",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos",
        "Bullous Disease Photos",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos",
        "Exanthems and Drug Eruptions",
        "Hair Loss Photos Alopecia and other Hair Diseases",
        "Herpes HPV and other STDs Photos",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Nail Fungus and other Nail Disease",
        "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Scabies Lyme Disease and other Infestations and Bites",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis Photos",
        "Warts Molluscum and other Viral Infections"
    ]

    top3_indices = prediction.argsort()[-3:][::-1]
    top_preds = [
        {
            "label": class_labels[i],
            "confidence": float(prediction[i])
        }
        for i in top3_indices
    ]

    metadata = {
        "age": age,
        "race": race,
        "gender": gender,
        "skin_color": skin_color,
        "skin_type": skin_type,
        "condition": condition_description
    }

    adjusted_preds = adjust_with_metadata(top_preds, metadata)

    return {
        "prediction": adjusted_preds,
        "metadata": metadata
    }
