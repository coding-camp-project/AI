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

# top-1 < 0.40 -> unknown, 0.40-0.60 -> warning, >= 0.60 -> clean
REJECT_THRESHOLD = 0.40
WARN_THRESHOLD = 0.60

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


# =========================================================
# MANUAL FOOD INPUT - tabel konversi, parser teks, search
# =========================================================
# Semua nilai nutrisi di CSV adalah per 100g. Nutrisi final dihitung:
#   nutrisi = nilai_csv * (gram / 100)

# Tabel konversi satuan -> gram (angka kasar, bisa disesuaikan)
UNIT_TO_GRAM = {
    "porsi": 150,
    "piring": 200,
    "mangkok": 200,
    "mangkuk": 200,
    "potong": 50,
    "buah": 100,
    "butir": 55,        # cth: telur
    "biji": 30,
    "lembar": 20,
    "iris": 25,
    "sendok makan": 15,
    "sdm": 15,
    "sendok teh": 5,
    "sdt": 5,
    "gelas": 240,
    "cangkir": 200,
    "centong": 100,
    "bungkus": 100,
    "tusuk": 30,        # cth: sate
    "gram": 1,
    "g": 1,
    "ons": 100,
    "kg": 1000,
}

# Default kalau satuan tidak dikenali / tidak ditulis
DEFAULT_GRAM = 100

# Kata bilangan -> angka (untuk teks "satu", "dua", dst)
WORD_TO_NUM = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    "setengah": 0.5, "seperempat": 0.25,
}


def parse_quantity(text):
    """Ekstrak jumlah (angka) dari potongan teks.
    Support: '2', '1/2', '0.5', 'dua', 'setengah'. Return float.
    """
    import re
    text = text.lower().strip()

    # Pecahan: 1/2, 3/4
    frac = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        return num / den if den != 0 else 1.0

    # Desimal / bulat: 1.5, 2
    dec = re.search(r'(\d+\.?\d*)', text)
    if dec:
        return float(dec.group(1))

    # Kata bilangan
    for word, num in WORD_TO_NUM.items():
        if word in text:
            return float(num)

    return 1.0  # default 1 kalau tidak ada angka


def parse_unit(text):
    """Deteksi satuan dari teks. Return (nama_satuan, gram_per_unit)."""
    text = text.lower()
    # Cek satuan multi-kata dulu (sendok makan sebelum 'sendok')
    for unit in sorted(UNIT_TO_GRAM.keys(), key=len, reverse=True):
        if unit in text:
            return unit, UNIT_TO_GRAM[unit]
    return None, DEFAULT_GRAM


def parse_food_text(text):
    """Pecah teks bebas jadi list item makanan.
    Input : 'nasi putih 1 porsi, ayam goreng 2 potong'
    Output: [
        {'raw': 'nasi putih 1 porsi', 'name_guess': 'nasi putih',
         'quantity': 1.0, 'unit': 'porsi', 'gram_per_unit': 150},
        ...
    ]
    """
    import re
    items = []
    # Pisah per item: koma, titik koma, atau ' dan '
    chunks = re.split(r'[,;]|\sdan\s', text)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        quantity = parse_quantity(chunk)
        unit, gram_per_unit = parse_unit(chunk)

        # Nama makanan = teks setelah buang angka, pecahan, dan satuan
        name = chunk.lower()
        name = re.sub(r'\d+\s*/\s*\d+', '', name)   # buang pecahan
        name = re.sub(r'\d+\.?\d*', '', name)        # buang angka
        # Buang kata bilangan (pakai word boundary)
        for w in WORD_TO_NUM:
            name = re.sub(r'\b' + re.escape(w) + r'\b', '', name)
        # Buang kata satuan (pakai word boundary biar 'g' tidak menghapus
        # huruf di tengah kata seperti 'goren[g]')
        for u in sorted(UNIT_TO_GRAM.keys(), key=len, reverse=True):
            name = re.sub(r'\b' + re.escape(u) + r'\b', '', name)
        name = re.sub(r'\s+', ' ', name).strip(' -.')

        items.append({
            "raw": chunk,
            "name_guess": name,
            "quantity": quantity,
            "unit": unit or "porsi",
            "gram_per_unit": gram_per_unit,
        })
    return items


def search_food_candidates(query, limit=5):
    """Cari kandidat makanan di CSV berdasarkan query teks.
    Return list nama makanan, urut dari paling relevan.
    Strategi: exact > startswith > contains semua kata > contains sebagian.
    """
    query = query.lower().strip()
    if not query:
        return []

    food_lower = nutrition_df["food_name"].str.lower().str.strip()
    query_words = query.split()

    scored = []
    for idx, fname in food_lower.items():
        score = 0
        if fname == query:
            score = 100
        elif fname.startswith(query):
            score = 80
        elif query in fname:
            score = 60
        elif all(w in fname for w in query_words):
            score = 40
        else:
            # hitung berapa kata query yang muncul
            matched = sum(1 for w in query_words if w in fname)
            if matched > 0:
                score = 10 * matched
        if score > 0:
            # makanan dengan nama lebih pendek = lebih relevan (tie-breaker)
            scored.append((score, -len(fname), nutrition_df.loc[idx, "food_name"]))

    scored.sort(reverse=True)
    return [name for _, _, name in scored[:limit]]


def calc_nutrition_scaled(food_name, grams):
    """Hitung nutrisi makanan untuk jumlah gram tertentu.
    CSV per 100g, jadi scaled = nilai_csv * (grams / 100).
    """
    base = get_nutrition(food_name)
    if base is None:
        return None
    factor = grams / 100.0
    return {k: round(v * factor, 2) for k, v in base.items()}


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
        "version": "3.0.0",
        "endpoints": {
            "POST /predict": "Analisis makanan: gambar, manual items, atau keduanya",
            "GET /search-food": "Cari kandidat makanan di database (untuk input manual)",
            "GET /health": "Health check",
            "GET /diseases": "List supported disease conditions",
            "GET /units": "List satuan porsi yang didukung"
        }
    }

@app.get("/diseases")
def list_diseases():
    return {
        "valid_diseases": sorted(VALID_DISEASES),
        "note": "Field 'disease' di /predict bersifat optional. Kalau kosong, recommendation umum."
    }

@app.get("/units")
def list_units():
    """List satuan porsi yang dikenali + konversi gram-nya."""
    return {
        "units": UNIT_TO_GRAM,
        "default_gram": DEFAULT_GRAM,
        "note": "Nilai konversi bersifat estimasi kasar."
    }


@app.get("/search-food")
def search_food(q: str, limit: int = 5):
    """Cari kandidat makanan di database CSV.

    Dipakai frontend saat user mengetik nama makanan di input manual.
    User ketik 'ayam goreng' -> dapat daftar kandidat -> user pilih satu.

    Query params:
    - q     : teks pencarian (nama makanan)
    - limit : jumlah kandidat maksimal (default 5)
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Parameter 'q' tidak boleh kosong.")

    limit = max(1, min(limit, 20))
    candidates = search_food_candidates(q.strip(), limit=limit)

    # Sertakan preview nutrisi per 100g untuk tiap kandidat
    results = []
    for name in candidates:
        nut = get_nutrition(name)
        results.append({
            "food_name": name,
            "nutrition_per_100g": nut,
        })

    return {
        "query": q.strip(),
        "count": len(results),
        "candidates": results,
    }


def _sum_nutrition(list_of_nutrition):
    """Jumlahkan beberapa dict nutrisi jadi satu total."""
    keys = ["calories", "protein", "fat", "carbohydrates", "sugar", "sodium", "fiber"]
    total = {k: 0.0 for k in keys}
    for nut in list_of_nutrition:
        if not nut:
            continue
        for k in keys:
            total[k] += nut.get(k, 0) or 0
    return {k: round(v, 2) for k, v in total.items()}


@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(default=None),
    disease: Optional[str] = Form(default=None),
    manual_items: Optional[str] = Form(default=None),
):
    """Analisis makanan. Fleksibel - terima salah satu / kombinasi:

    Form fields:
    - image        (file, opsional)  : foto makanan
    - manual_items (str, opsional)   : JSON list item makanan yang sudah dipilih
                                       user dari /search-food. Format tiap item:
                                       {"food_name": "...", "quantity": 1.5, "unit": "porsi"}
    - disease      (str, opsional)   : riwayat penyakit untuk recommendation

    Skenario:
    1. image saja            -> deteksi model
    2. manual_items saja     -> hitung nutrisi dari item yang dipilih
    3. image + manual_items  -> gabung: hasil gambar + item manual
    """
    # --- Validasi disease ---
    if disease is not None and disease.strip() != "":
        disease = disease.strip().lower()
        if disease not in VALID_DISEASES:
            raise HTTPException(
                status_code=400,
                detail=f"Disease tidak valid. Pilihan: {sorted(VALID_DISEASES)}"
            )
    else:
        disease = None

    # --- Parse manual_items (JSON) ---
    parsed_manual = []
    if manual_items and manual_items.strip():
        try:
            parsed_manual = json.loads(manual_items)
            if not isinstance(parsed_manual, list):
                raise ValueError("manual_items harus berupa list")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Format manual_items tidak valid: {e}. "
                       f"Harus JSON list of objects."
            )

    # Minimal salah satu harus ada
    has_image = image is not None and image.filename
    if not has_image and not parsed_manual:
        raise HTTPException(
            status_code=400,
            detail="Minimal salah satu wajib diisi: 'image' atau 'manual_items'."
        )

    try:
        result = {"success": True}
        nutrition_parts = []   # kumpulan nutrisi untuk grand total
        primary_food_name = None   # untuk recommendation

        # =================================================
        # BAGIAN 1: GAMBAR (kalau ada)
        # =================================================
        image_result = None
        if has_image:
            if not image.content_type or not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File harus berupa gambar.")

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
                {"food_name": class_names[idx],
                 "confidence_score": round(float(predictions[idx]), 4)}
                for idx in top_indices
            ]
            best_food = class_names[top_indices[0]]
            best_confidence = float(predictions[top_indices[0]])

            if best_confidence < REJECT_THRESHOLD:
                # Gambar tidak dikenali
                image_result = {
                    "recognized": False,
                    "message": "Gambar tidak terdeteksi sebagai makanan yang dikenali sistem.",
                    "suggestion": "Gunakan input manual untuk makanan ini, atau foto lebih jelas.",
                    "top_confidence": round(best_confidence, 4),
                }
                # Kalau tidak ada manual_items, ini benar-benar gagal
                if not parsed_manual:
                    return {
                        "success": False,
                        "error": "unknown_food",
                        "message": "Gambar tidak dikenali dan tidak ada input manual.",
                        "suggestion": "Tambahkan input manual makanan, atau foto lebih jelas.",
                        "image_result": image_result,
                    }
            else:
                nut = get_nutrition(best_food)
                image_result = {
                    "recognized": True,
                    "best_prediction": {
                        "food_name": best_food,
                        "confidence_score": round(best_confidence, 4),
                    },
                    "top_predictions": top_predictions,
                    "nutrition": nut,
                    "warning": (
                        "Model kurang yakin terhadap gambar."
                        if best_confidence < WARN_THRESHOLD else None
                    ),
                }
                if nut:
                    nutrition_parts.append(nut)
                primary_food_name = best_food

        # =================================================
        # BAGIAN 2: MANUAL ITEMS (kalau ada)
        # =================================================
        manual_result = []
        if parsed_manual:
            for item in parsed_manual:
                food_name = (item.get("food_name") or "").strip()
                quantity = item.get("quantity", 1)
                unit = (item.get("unit") or "porsi").strip().lower()

                if not food_name:
                    continue

                try:
                    quantity = float(quantity)
                except (TypeError, ValueError):
                    quantity = 1.0

                # Konversi ke gram
                gram_per_unit = UNIT_TO_GRAM.get(unit, DEFAULT_GRAM)
                total_gram = quantity * gram_per_unit

                # Hitung nutrisi
                scaled = calc_nutrition_scaled(food_name, total_gram)

                item_result = {
                    "food_name": food_name,
                    "quantity": quantity,
                    "unit": unit,
                    "total_gram": round(total_gram, 1),
                    "nutrition": scaled,
                }
                if scaled is None:
                    item_result["error"] = (
                        f"'{food_name}' tidak ditemukan di database. "
                        f"Pastikan nama persis sesuai hasil /search-food."
                    )
                else:
                    nutrition_parts.append(scaled)
                    if primary_food_name is None:
                        primary_food_name = food_name

                manual_result.append(item_result)


        grand_total = _sum_nutrition(nutrition_parts)

        # Recommendation berdasarkan grand total (pakai nama "makanan gabungan"
        # kalau lebih dari 1 sumber)
        n_sources = len(nutrition_parts)
        if n_sources == 0:
            recommendation = "Tidak ada data nutrisi yang bisa dianalisis."
        elif n_sources == 1 and primary_food_name:
            recommendation = generate_recommendation(
                primary_food_name, grand_total, disease=disease
            )
        else:
            # Banyak item - kasih rekomendasi untuk total gabungan
            recommendation = generate_recommendation(
                "kombinasi makanan", grand_total, disease=disease
            )

        result["image_result"] = image_result
        result["manual_items"] = manual_result if parsed_manual else None
        result["grand_total_nutrition"] = grand_total
        result["recommendation"] = recommendation
        result["sources_count"] = n_sources

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))