"""
<>Konfig untuk logging<>
"""

import logging
import sys

from app.core.config import settings

def setup_logging() -> None:
    # buat cetak agar tau apa aja yang bakal dicetak
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
# buat setiap yang panggil pake ini
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)