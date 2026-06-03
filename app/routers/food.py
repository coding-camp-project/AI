"""
    ini endpoint GET khusus bahas food 
    
    Endpoint:
    1. GET /search-food : cari kandidat makanan di csv
    2. GET /units : cari daftar satuan
    3. GET /disease : cari penyakit yang bisa di track tersedia
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.dependencies import get_nutrition_service
from app.services.nutrition_service import NutritionService, UNIT_TO_GRAM, DEFAULT_GRAM
from app.services.recommendation_service import VALID_DISEASES
from app.schemas.schema import SearchFoodResponse

# api router sub
router = APIRouter(tags=['Food Data'])

@router.get('/search-food', response_model=SearchFoodResponse)
def search_food(
    q: str = Query(..., description="Teks pencarian nama makanan", min_length=1),
    limit: int = Query(default=5, ge=1, le=20, description="Jumlah Kandidat"),
    nutrition: NutritionService = Depends(get_nutrition_service)
    
):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Parameter 'q' tidak boleh kosong.")
    candidate_names = nutrition.search_candidates(query, limit=limit)
    
    candidates = [
        {
            "food_name": name,
            "nutrition_per_100g": nutrition.get_nutrition(name)
        }
        for name in candidate_names
    ]
    
    return {
        "query": query,
        "count": len(candidates),
        "candidates": candidates
    }

@router.get("/units")
def list_units():
    return {
        "units": UNIT_TO_GRAM,
        "default_gram": DEFAULT_GRAM
    }

@router.get("/disease")
def list_disease():
    return {
        "disease": sorted(VALID_DISEASES),
    }