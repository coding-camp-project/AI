"""
<> KONFIGURASI aplikasi terpusat <>
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):

    # <> Info Aplikasi <>
    APP_NAME: str
    APP_VERSION: str
    APP_DESCRIPTION: str

    # <> File Path <>
    MODEL_PATH: str
    CLASS_NAMES_PATH: str
    DATASETS: str

    # <> CNN Config <>
    IMG_SIZE: int

    REJECT_THRESHOLD: float
    WARN_THRESHOLD: float

    TEMPERATURE_SCALE: float

    # <> CORS <>
    CORS: str

    # <> Server <>
    HOST: str
    PORT: int

    # <> Logging <>
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

        return [
            origin.strip()
            for origin in self.CORS.split(",")
            if origin.strip()
        ]


settings = Settings()