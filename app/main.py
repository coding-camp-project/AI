"""
    jalaninya pake python -m uvicorn app.main:app --reload
    
    gabungin 
    1. setup logging
    2. setup CORS
    3. laod services
    4. daftar semua router
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.services.dependencies import init_services
from app.routers import check_system_api, predict, food, chat_route

setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        lifecycle kode sebelum dijalankan startup dan setelah dijalankan shutdown
    """
    
    # Startup
    logger.info("Memulai %s v%s", settings.APP_NAME, settings.APP_VERSION)
    init_services() # laod model ama database
    logger.info("Aplikasi telah siap!")
    yield
    # Shutdown
    logger.info("Aplikasi shutting down")

# Buat aplikasi FASTAPI
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Daftartkan semua rotuer
app.include_router(check_system_api.router)
app.include_router(food.router)
app.include_router(predict.router)
app.include_router(chat_route.router)
