"""
    hanya check by ricek system api
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=['system'])

@router.get("/")
def root():
    return {
        "message": f"{settings.APP_NAME} is Runnig!",
        "version": settings.APP_VERSION,
        "docs": "check dokumen in endpoint /docs"
    }

@router.get("/health-check")
def health_check():
    return {"status": "Wokeh!"}
