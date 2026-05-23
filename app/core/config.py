"""
<> KONFIGURASI aplikasi terpusat <>
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # <> Info Aplikasi <>
    APP_NAME: str = "Nutrify AI - API"
    APP_VERSION: str = "1.0"
    APP_DESCRIPTION: str = "Indonesian Food Recognation and Recomendation For Disease"
    
    MODEL_PATH: str = "nutrify_model.keras"
    CLASS_NAMES_PATH: str = "class_names.json"
    DATASETS: str = "indonesian_food_clean.csv"
    
    IMG_SIZE: int = 224
    
    REJECT_THRESHOLD: float = 0.61
    WARN_THRESHOLD: float = 0.62
    
    TEMPERATURE_SCALE: float = 5.0
    
    CORS: str = "*"
    
    HOST: str = "0.0.0.0"
    PORT: int = 8173
    
    LOG_LEVEL: str = "INFO"
    
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