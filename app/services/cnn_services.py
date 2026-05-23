"""
Service yang khusus buat Deep learning

Tanggung jawab:
1. Load model (cuman sekali, saat start)
2. Load daftar nama kelas yang bentuknya json
3. Prerpoces gambar
4. prediksi makanan dari gambar
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
    
    # ini buat load si model di route
    def __init__(self):
        logger.info("Tunggu AI Model...")
        self.model = tf.keras.models.load_model(
            settings.MODEL_PATH,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
        logger.info("Model berhasil di dapat")    
        # Ini buat load nama kelas di json
        with open(settings.CLASS_NAMES_PATH, 'r') as f:
            self.class_names = json.load(f)
        logger.info("Mendapatkan %d class", len(self.class_names))
        # Nentuin temparture scale
        self.temperature = self._detect_temperature()
    
    
    def _detect_temperature(self) -> float:
        # Disini model yang butuh scale cuman L2NormalizationLayer
        
        layer_names = [type(layer).__name__ for layer in self.model.layers]
        has_l2norm = "L2NormalizationLayer" in layer_names
        has_scaled = "ScaledDenseLayer" in layer_names
        
        if has_scaled and not has_l2norm:
            logger.info("ScaledDenseLayer detected -> temperature = 1.0")
            return 1.0
        elif has_l2norm:
            logger.info("L2NormalizationLayer detected -> temperature = %.1f", settings.TEMPERATURE_SCALE)
            return settings.TEMPERATURE_SCALE
        else:
            logger.warning("Unkown architecture -> temperature = %.1f", settings.TEMPERATURE_SCALE)
            return settings.TEMPERATURE_SCALE
        
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
            Ubah gambar pil jadi input model
            sebenarnya langkahnya itu convert RGB -> resize -> preprocessing efficientnet -> tambah dimensi batch
        """
        image = image.convert('RGB')
        image = image.resize((settings.IMG_SIZE, settings.IMG_SIZE))
        arr = np.array(image).astype(np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
    
    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """
            prediksi maknaan dari bytes gambar per
            nanti returnya: 
            1. best_food, best_confidence
        """
        pil_image = Image.open(io.BytesIO(image_bytes))
        processed = self.preprocess(pil_image)
        
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # ini buat scale pake default temp
        if self.temperature != 1.0:
            predictions = self._apply_temperature(predictions)
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            {
                "food_name": self.class_names[idx],
                "confidence_score": round(float(predictions[idx]), 4)
            }
            for idx in top_indices
        ]
        
        return {
            "best_food": self.class_names[top_indices[0]],
            "best_confidence": float(predictions[top_indices[0]]),
        }

    def _apply_temperature(self, predictions: np.ndarray) -> np.ndarray:
        """
            buat re-scale distribusi probab pake temperature tadi
        """
        log_probs = np.log(predictions + 1e-9)
        scaled = log_probs / self.temperature #diganti
        scaled = scaled - scaled.max()
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum()
        
    
    
