"""
Service yang khusus buat Deep learning
"""
import io
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.cnn_layers import CUSTOM_OBJECTS

logger = get_logger(__name__)


class CNNService:

    def __init__(self):
        logger.info("Tunggu AI Model...")
        full_model = tf.keras.models.load_model(
            settings.MODEL_PATH,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
        logger.info("Model berhasil di dapat")

        # Bikin model yang outputnya LOGITS (sebelum Softmax), bukan probability.
        # Arsitektur notebook: ... -> ScaledDenseLayer -> Softmax
        # Layer kedua-dari-terakhir = ScaledDenseLayer = logits.
        self.model = self._build_logits_model(full_model)

        with open(settings.CLASS_NAMES_PATH, 'r') as f:
            self.class_names = json.load(f)
        logger.info("Mendapatkan %d class", len(self.class_names))

    def _build_logits_model(self, full_model: tf.keras.Model) -> tf.keras.Model:
        """Potong model di layer sebelum Softmax supaya dapat logits mentah.
        Logits jauh lebih bagus buat temperature scaling daripada
        log(probability) yang sudah jenuh."""
        last = full_model.layers[-1]
        if type(last).__name__ in ("Softmax", "Activation"):
            logits_model = tf.keras.Model(
                inputs=full_model.input,
                outputs=full_model.layers[-2].output,
            )
            logger.info("Logits model dibangun (output = %s)",
                        type(full_model.layers[-2]).__name__)
            return logits_model
        else:
            logger.warning("Layer terakhir bukan Softmax, pakai model apa adanya")
            return full_model

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert('RGB')
        image = image.resize((settings.IMG_SIZE, settings.IMG_SIZE))
        arr = np.array(image).astype(np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax dengan temperature. T > 1 -> lebih lembut (kurang pede)."""
        x = x / temperature
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def predict(self, image_bytes: bytes) -> dict:
        pil_image = Image.open(io.BytesIO(image_bytes))
        processed = self.preprocess(pil_image)

        logits = self.model.predict(processed, verbose=0)[0]
        probs = self._softmax(logits, temperature=settings.TEMPERATURE_SCALE)

        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])

        # 3 tingkat berdasarkan threshold di .env
        if best_conf < settings.REJECT_THRESHOLD:
            status = "rejected"        # kemungkinan bukan makanan yang dikenali
        elif best_conf < settings.WARN_THRESHOLD:
            status = "uncertain"       # ragu, tampilkan dengan peringatan
        else:
            status = "confident"

        return {
            "best_food": self.class_names[best_idx] if status != "rejected" else None,
            "best_confidence": best_conf,
            "status": status,
            "raw_prediction": self.class_names[best_idx],
        }