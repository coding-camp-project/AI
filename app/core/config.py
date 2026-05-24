"""
<> KONFIGURASI aplikasi terpusat <>
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # <> Info Aplikasi <>
    APP_NAME: str
    APP_VERSION: str
    APP_DESCRIPTION: str
    
    MODEL_PATH: str
    CLASS_NAMES_PATH: str
    DATASETS: str
    
    IMG_SIZE: int
    
    REJECT_THRESHOLD: float
    WARN_THRESHOLD: float
    
    TEMPERATURE_SCALE: float
    
    CORS: str 
    
    HOST: str 
    PORT: int 
    
    LOG_LEVEL: str 
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS.strip() == "*":
            return ["*"]

        return [o.strip() for o in self.CORS.split(",") if o.strip()]

settings = Settings()