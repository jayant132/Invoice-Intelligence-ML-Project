"""OCR engine interface and implementations."""

from .base import OCREngine, OCRLine
from .factory import get_ocr_engine

__all__ = ["OCREngine", "OCRLine", "get_ocr_engine"]
