"""
    Ini adalah service yang berhubungan sama nutrisi (perhitungan gram, load csv, dan cari makanan di csv)
"""

import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

UNIT_TO_GRAM = {
    "porsi": 150,
    "piring": 200,
    "mangkok": 200,
    "mangkuk": 200,
    "potong": 50,
    "buah": 100,
    "butir": 55,
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
    "tusuk": 30,
    "gram": 1,
    "g": 1,
    "ons": 100,
    "kg": 1000,
}

NUTRITION_KEYS = ['calories', 'protein', 'fat', 'carbohydrates', 'sugar', 'sodium', 'fiber']
# Nutrisi 
DEFAULT_GRAM = 100

class NutritionService:
    """Manage nutrisi"""
    
    def __init__(self):
        logger.info("Tunggu nutrisi database...")
        self.df = pd.read_csv(settings.DATASETS)
        self._food_lower = self.df['food_name'].str.lower().str.strip()
        logger.info("Nutrisi database berhasil (%d makanan)", len(self.df))

    def get_nutrition(self, food_name: str) -> dict | None:
        normalized = food_name.replace("_", " ").lower().strip()
        result = self.df[self._food_lower == normalized]
        if result.empty:
            return None
        row = result.iloc[0]
        return {key: float(row[key]) for key in NUTRITION_KEYS}

    def search_candidates(self, query: str, limit: int = 5) -> list[str]:
        """
            buat cari makanan  yang cocok 
            skor: exact > startstwith > contains > kata sebagian
        """
        query = query.lower().strip()
        if not query:
            return []
        
        query_words = query.split()
        scored = []
        
        for idx, fname in self._food_lower.items():
            score = 0
            if fname == query:
                score = 100
            elif fname.startswith(query):
                score = 80
            elif query in fname:
                score = 60
            elif all(w in fname for w in query_words):
                score= 40
            else:
                matched = sum(1 for w in query_words if w in fname)
                if matched > 0:
                    score = 10 * matched
                
            
            if score > 0:
                # nama pendek = lebih relevan
                scored.append((score, -len(fname), self.df.loc[idx, 'food_name']))
        
        scored.sort(reverse=True)
        return [name for _, _, name in scored[:limit]]

    # HItung nutrisi berdasarkan gram (rumus)
    def calc_scaled(self, food_name: str, grams: float) -> dict | None:
        base = self.get_nutrition(food_name)
        if base is None:
            return None
        factor = grams / DEFAULT_GRAM
        return {k: round(v * factor, 2) for k, v in base.items()}
        
                
    # Konversi satuan unit
    @staticmethod
    def unit_to_gram(unit: str) -> int:
        return UNIT_TO_GRAM.get(unit.strip().lower(), DEFAULT_GRAM)
    
    # Jumlahkan Nutrisinya
    @staticmethod
    def sum_nutrition(list_of_nutrition: list[dict]) -> dict:
        total = {k: 0.0 for k in NUTRITION_KEYS}
        for nut in list_of_nutrition:
            if not nut:
                continue
            for k in NUTRITION_KEYS:
                total[k] += nut.get(k, 0) or 0
        
        return {k: round(v, 2) for k, v in total.items()}
