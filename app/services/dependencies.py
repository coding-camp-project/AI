"""
    Dependency untuk fastapi
    
    Service (ML & Nutrition) di-load sekali saja saat aplikasi start,
lalu instance yang sama dipakai di semua request. Ini penting karena
load model TF itu berat - tidak boleh diulang tiap request.

Cara kerja: fungsi get_xxx_service() dipanggil FastAPI saat ada request,
tapi karena objeknya disimpan di variabel modul, yang dikembalikan
selalu instance yang sama.
"""

from app.core.logging_config import get_logger
from app.services.cnn_services import CNNService
from app.services.nutrition_service import NutritionService

logger = get_logger(__name__)

# ini buat nampung instance dari luar
_cnn_service: CNNService | None = None
_nutrition_service: NutritionService | None = None

# Fungsi pertama init dari service
def init_services() -> None:
    """Inisialisasi buat di panggil sekali biar gak berat"""
    global _cnn_service, _nutrition_service
    logger.info("Initializing services...")
    _nutrition_service = NutritionService()
    _cnn_service = CNNService()
    logger.info("Semua service sudah siap!")
    

def get_cnn_service() -> CNNService:
    """ini buat ambil sebagai depen aja di router"""
    if _cnn_service is None:
        raise RuntimeError("CNNService belum diinisialisasi. panggil init_services() dulu.")
    return _cnn_service

def get_nutrition_service() -> NutritionService:
    """sama kayak diatas"""
    if _nutrition_service is None:
        raise RuntimeError("Nutrition Service belum diinisialisasi. panggil init_services() dulu.")
    return _nutrition_service

