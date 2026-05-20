import keras
from keras.layers import Dense

original_dense_init = Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    original_dense_init(self, *args, **kwargs)

Dense.__init__ = patched_dense_init


import io
import json
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Nutrify API",
    description="Indonesian Food Recognition API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224
MODEL_PATH = "nutrify_model.keras"
CLASS_NAMES_PATH = "class_names.json"
NUTRITION_CSV = "indonesian_food_clean.csv"

# top-1 < 0.10 -> unknown, 0.10-0.40 -> warning, >= 0.40 -> clean
REJECT_THRESHOLD = 0.10
WARN_THRESHOLD = 0.40

# L2Norm di arsitektur bikin softmax flat (~4-10%), temperature re-scale supaya distribusi realistis
TEMPERATURE = 5.0

VALID_DISEASES = {"obesitas", "diabetes", "hipertensi", "asam_urat", "kolesterol"}


@tf.keras.utils.register_keras_serializable()
class L2NormalizationLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class ScaledDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units, initial_scale=10.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        logits = tf.matmul(inputs, self.kernel) + self.bias
        return logits * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'initial_scale': self.initial_scale,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, alpha=0.25, num_classes=25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true = tf.cast(
            tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes),
            tf.float32
        )
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "num_classes": self.num_classes
        })
        return config


print("Loading Nutrify model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "L2NormalizationLayer": L2NormalizationLayer,
        "ScaledDenseLayer": ScaledDenseLayer,
        "FocalLoss": FocalLoss,
    },
    compile=False
)

print("Model loaded successfully!")

_layer_names = [type(l).__name__ for l in model.layers]
_has_l2norm = 'L2NormalizationLayer' in _layer_names
_has_scaled = 'ScaledDenseLayer' in _layer_names

if _has_scaled and not _has_l2norm:
    print(f"ScaledDenseLayer detected -> TEMPERATURE set to 1.0")
    TEMPERATURE = 1.0
elif _has_l2norm:
    print(f"L2NormalizationLayer detected -> TEMPERATURE = {TEMPERATURE}")
else:
    print(f"Unknown architecture -> TEMPERATURE = {TEMPERATURE}")


with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} classes")

nutrition_df = pd.read_csv(NUTRITION_CSV)
print("Nutrition data loaded!")


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def get_nutrition(food_name):
    normalized_food = food_name.replace("_", " ").lower().strip()
    result = nutrition_df[
        nutrition_df["food_name"].str.lower().str.strip() == normalized_food
    ]
    if result.empty:
        return None
    row = result.iloc[0]
    return {
        "calories": float(row["calories"]),
        "protein": float(row["protein"]),
        "fat": float(row["fat"]),
        "carbohydrates": float(row["carbohydrates"]),
        "sugar": float(row["sugar"]),
        "sodium": float(row["sodium"]),
        "fiber": float(row["fiber"])
    }


def _check_diabetes(nutrition):
    issues = []
    if nutrition["sugar"] >= 15:
        issues.append(f"kandungan gula {nutrition['sugar']:.1f}g tergolong sangat tinggi")
    elif nutrition["sugar"] >= 8:
        issues.append(f"kandungan gula {nutrition['sugar']:.1f}g tergolong cukup tinggi")
    if nutrition["carbohydrates"] >= 40:
        issues.append(f"karbohidrat {nutrition['carbohydrates']:.1f}g cukup tinggi")
    return issues

def _check_hipertensi(nutrition):
    issues = []
    sodium = nutrition["sodium"]
    if sodium >= 600:
        issues.append(f"sodium {sodium:.0f}mg sangat tinggi (batas WHO 2000mg/hari)")
    elif sodium >= 400:
        issues.append(f"sodium {sodium:.0f}mg cukup tinggi")
    return issues

def _check_obesitas(nutrition):
    issues = []
    if nutrition["calories"] >= 300:
        issues.append(f"kalori {nutrition['calories']:.0f} kcal tergolong tinggi")
    if nutrition["fat"] >= 15:
        issues.append(f"lemak {nutrition['fat']:.1f}g tergolong tinggi")
    if nutrition["sugar"] >= 15:
        issues.append(f"gula {nutrition['sugar']:.1f}g tinggi")
    return issues

def _check_asam_urat(nutrition, food_name):
    issues = []
    name_lower = food_name.lower().replace("_", " ")
    high_purin_keywords = [
        "daging", "rawon", "sate", "burger sapi", "ikan",
        "seafood", "udang", "kerang", "jeroan", "hati", "usus"
    ]
    for kw in high_purin_keywords:
        if kw in name_lower:
            issues.append(f"makanan ini tergolong tinggi purin (mengandung {kw})")
            break
    if nutrition["protein"] >= 20:
        issues.append(f"protein hewani {nutrition['protein']:.1f}g cukup tinggi")
    return issues

def _check_kolesterol(nutrition, food_name):
    issues = []
    name_lower = food_name.lower().replace("_", " ")
    if nutrition["fat"] >= 15:
        issues.append(f"lemak {nutrition['fat']:.1f}g tergolong tinggi")
    fried_keywords = ["goreng", "kentang_goreng", "nugget", "martabak", "burger"]
    for kw in fried_keywords:
        if kw in name_lower:
            issues.append("termasuk makanan tinggi lemak jenuh / digoreng")
            break
    return issues


def generate_recommendation(food_name, nutrition, disease=None):
    if nutrition is None:
        return "Data nutrisi tidak tersedia untuk makanan ini."

    display_name = food_name.replace("_", " ").title()
    cal = nutrition["calories"]
    protein = nutrition["protein"]
    fat = nutrition["fat"]

    if disease and disease in VALID_DISEASES:
        if disease == "diabetes":
            issues = _check_diabetes(nutrition)
            disease_label = "diabetes"
        elif disease == "hipertensi":
            issues = _check_hipertensi(nutrition)
            disease_label = "hipertensi"
        elif disease == "obesitas":
            issues = _check_obesitas(nutrition)
            disease_label = "obesitas / manajemen berat badan"
        elif disease == "asam_urat":
            issues = _check_asam_urat(nutrition, food_name)
            disease_label = "asam urat"
        elif disease == "kolesterol":
            issues = _check_kolesterol(nutrition, food_name)
            disease_label = "kolesterol tinggi"

        if not issues:
            return (
                f"{display_name} relatif aman untuk kondisi {disease_label} Anda. "
                f"Per 100g: {cal:.0f} kcal, protein {protein:.1f}g, lemak {fat:.1f}g. "
                f"Tetap perhatikan porsi konsumsi total harian."
            )
        elif len(issues) == 1:
            return (
                f"Untuk penderita {disease_label}, {display_name} sebaiknya dikonsumsi dengan hati-hati "
                f"karena {issues[0]}. Disarankan kurangi porsi atau cari alternatif."
            )
        else:
            return (
                f"Untuk penderita {disease_label}, {display_name} sebaiknya dibatasi atau dihindari karena: "
                f"{', '.join(issues)}. Konsultasikan dengan ahli gizi untuk pengaturan diet yang tepat."
            )

    if cal < 150:
        category = "rendah kalori dan cocok untuk diet"
    elif cal < 300:
        category = "cukup seimbang untuk konsumsi harian"
    else:
        category = "tinggi kalori, sebaiknya dikonsumsi dengan porsi terkontrol"

    protein_text = (
        "cukup baik untuk mendukung pembentukan otot."
        if protein > 15
        else "perlu dilengkapi sumber protein lain."
    )

    fat_text = (
        "tergolong tinggi, perhatikan frekuensi konsumsi."
        if fat > 15
        else "masih dalam batas wajar."
    )

    return (
        f"{display_name} termasuk makanan {category} "
        f"dengan {cal:.0f} kcal per 100g sajian. "
        f"Kandungan protein {protein:.1f}g {protein_text} "
        f"Lemak sebesar {fat:.1f}g {fat_text}"
    )


@app.get("/")
def home():
    return {
        "message": "Nutrify API is running!",
        "version": "2.0.0",
        "endpoints": {
            "POST /predict": "Upload food image (multipart/form-data: image + optional disease)",
            "GET /health": "Health check",
            "GET /diseases": "List supported disease conditions"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/diseases")
def list_diseases():
    return {
        "valid_diseases": sorted(VALID_DISEASES),
        "note": "Field 'disease' di /predict bersifat optional. Kalau kosong, recommendation umum."
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    disease: Optional[str] = Form(default=None),
):
    if disease is not None and disease.strip() != "":
        disease = disease.strip().lower()
        if disease not in VALID_DISEASES:
            raise HTTPException(
                status_code=400,
                detail=f"Disease tidak valid. Pilihan: {sorted(VALID_DISEASES)}"
            )
    else:
        disease = None

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diupload harus gambar.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(pil_image)

        predictions = model.predict(processed_image, verbose=0)[0]

        if TEMPERATURE != 1.0:
            log_probs = np.log(predictions + 1e-9)
            scaled_log_probs = log_probs * TEMPERATURE
            scaled_log_probs = scaled_log_probs - scaled_log_probs.max()
            exp_scaled = np.exp(scaled_log_probs)
            predictions = exp_scaled / exp_scaled.sum()

        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            {"food_name": class_names[idx], "confidence_score": round(float(predictions[idx]), 4)}
            for idx in top_indices
        ]

        best_food = class_names[top_indices[0]]
        best_confidence = float(predictions[top_indices[0]])

        if best_confidence < REJECT_THRESHOLD:
            return {
                "success": False,
                "error": "unknown_food",
                "message": "Gambar tidak terdeteksi sebagai makanan Indonesia yang dikenali sistem.",
                "suggestion": "Pastikan gambar berisi makanan dengan pencahayaan cukup. Sistem hanya mendukung 25 jenis makanan Indonesia.",
                "debug_info": {
                    "top_confidence": round(best_confidence, 4),
                    "threshold": REJECT_THRESHOLD
                }
            }

        nutrition = get_nutrition(best_food)
        recommendation = generate_recommendation(best_food, nutrition, disease=disease)

        warning = None
        if best_confidence < WARN_THRESHOLD:
            warning = (
                "Model kurang yakin terhadap gambar. "
                "Coba gunakan foto yang lebih jelas dengan fokus ke satu makanan di tengah."
            )

        return {
            "success": True,
            "best_prediction": {
                "food_name": best_food,
                "confidence_score": round(best_confidence, 4)
            },
            "top_predictions": top_predictions,
            "nutrition": nutrition,
            "recommendation": recommendation,
            "warning": warning
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
