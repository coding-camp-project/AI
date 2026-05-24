"""
    Ini adalah komponen utama dari router yaitu endpoint /predict
    
    dia berbentuk POST
    Fleksibel - menerima
    1. Gambar saja -> deteksi via model bentuk upload
    2. Manual items ketik -> hitung nutrisi dari item yang dipilih tapi formatnya wajib json [{"food_name": "Nasi Goreng", "quantity": 3, "units": "porsi"}, ....]
    3. Gambar + manual -> combine keduanya
"""

import json

from fastapi import APIRouter, HTTPException, Depends, File, Form, UploadFile

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.dependencies import get_cnn_service, get_nutrition_service
from app.services.cnn_services import CNNService
from app.services.nutrition_service import NutritionService
from app.services.recomendation_service import generate_recomendation, VALID_DISEASES

logger = get_logger(__name__)
router = APIRouter(tags=["Predictions"])

def _parse_manual_items(raw: str | None) -> list[dict]:
    if raw is None:
        return []
    
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("string", "null", "none"):
        return []

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Field 'manual_items' harus berupa JSON array. Contoh: "
                '[{"food_name":"nasi putih","quantity":1,"unit":"porsi"},  ]. '
            )
        )
    
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=400,
            detail="field 'manual_items' harus berupa JSON array (list.)"
        )
    return parsed

def _validate_disease(disease: str | None) -> str | None:
    if disease is None or disease.strip() == "":
        return None
    disease = disease.strip().lower()
    if disease not in VALID_DISEASES:
        raise HTTPException(
            status_code=400,
            detail=f"Disease tidak valid, Pilihan: {sorted(VALID_DISEASES)}"
        )
    return disease

@router.post("/predict")
async def predict(
    image: UploadFile | None = File(default=None),
    disease: str | None = Form(default=None),
    manual_items: str | None = Form(default=None),
    cnn: CNNService = Depends(get_cnn_service),
    nutrition: NutritionService = Depends(get_nutrition_service)
):
    """
        Analisis makanan gambar dan/atau input manual
        form fields:
        - image : foto makanan 
        - manual_items: json array item makanan
        - disease : riwayat penyakit wajib buat rekomendasi
    """
    disease = _validate_disease(disease)
    parsed_manual = _parse_manual_items(manual_items)
    
    has_image = image is not None and image.filename
    if not has_image and not parsed_manual:
        raise HTTPException(status_code=400,
                            detail="Minimal salah satu wajib diisi: 'image' atau 'manual_items'")
    
    try:
        nutrition_parts: list[dict] = []
        primary_food: str | None = None
        image_result = None
        manual_result = []
        
        # Gambar
        if has_image:
            if not image.content_type or not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File harus berupa gambar.")

            image_bytes = await image.read()
            pred = cnn.predict(image_bytes)
            
            if pred["best_confidence"] < settings.REJECT_THRESHOLD:
                # Gambar gak kenal
                image_result = {
                    "recognized": False,
                    "message": "Gambar tidak terdeteksi sebagai makanan yang dikenali",
                    "suggestion": "Gunakan input foto lebih jelas",
                    "confidence": round(pred["best_confidence"], 4)
                    
                }
                
                if not parsed_manual:
                    return {
                        "succes": False,
                        "error": "Unkown_food",
                        "message": "Gambar tidak dikenali dan tidak ada input manual.",
                        "image_result": image_result
                    }
            else:
                nut = nutrition.get_nutrition(pred['best_food'])
                image_result = {
                    "recognized": True,
                    "best_prediction": {
                        "food_name": pred["best_food"],
                        "confidence_score": round(pred['best_confidence'], 4)
                    },
                    "nutrition": nut,
                    "warning": (
                        "Model kurang yakin terhadap gambar."
                        if pred['best_confidence'] < settings.WARN_THRESHOLD
                        else None
                    )
                }
                if nut:
                    nutrition_parts.append(nut)
                primary_food = pred['best_food']
        
        # Manual items
        for item in parsed_manual:
            food_name = (item.get('food_name') or "").strip()
            if not food_name:
                continue
                
            try:
                quantity = float(item.get('quantity', 1))
            except (TypeError, ValueError):
                quantity = 1.0
            
            unit = (item.get("unit") or "porsi").strip().lower()
            gram_per_unit = nutrition.unit_to_gram(unit)
            total_gram = quantity * gram_per_unit
            
            scaled = nutrition.calc_scaled(food_name, total_gram)
            
            item_result = {
                "food_name": food_name,
                "quantity": quantity,
                "unit": unit,
                "total_gram": round(total_gram, 1),
                "nutrition": scaled
            }
            
            if scaled is None:
                item_result["error"] = (
                    f"'{food_name}' tidak ditemukan"
                    f"Pastikan nama persis sesuai hasil dari pencarian."
                )
            else:
                nutrition_parts.append(scaled)
                if primary_food is None:
                    primary_food = food_name
            
            manual_result.append(item_result)

        # Gabungan
        grand_total = nutrition.sum_nutrition(nutrition_parts)
        n_sources = len(nutrition_parts)
        
        if n_sources == 0:
            recomendation = "Tidak ada data nutrisi yang bisa dianalisis"
        elif n_sources == 1 and primary_food:
            recomendation = generate_recomendation(primary_food, grand_total, disease)
        else:
            recomendation = generate_recomendation("Kombinasi makanan", grand_total, disease)
        
        return {
            "sucess": True,
            "image_result": image_result,
            "manual_items": manual_result if parsed_manual else None,
            "grand_total_nutrition": grand_total,
            "recommendation": recomendation,
            "sources_count": n_sources
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error saat memproses /predict")
        raise HTTPException(status_code=500, detail=str(e))