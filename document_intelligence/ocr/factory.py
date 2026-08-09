"""Factory that selects the best available OCR engine."""

from .base import OCREngine


def get_ocr_engine(preference: str = "auto") -> OCREngine:
    """Return an OCR engine.

    Args:
        preference: One of "auto" (default), "paddle" or "tesseract".
            "auto" tries PaddleOCR first and falls back to Tesseract.

    Raises:
        ValueError: If an unknown engine name is requested.
        RuntimeError: If no engine can be initialised.
    """
    engines = {
        "paddle": _paddle_engine,
        "tesseract": _tesseract_engine,
    }
    if preference not in ("auto", *engines):
        raise ValueError(f"Unknown OCR engine: {preference!r}")

    if preference != "auto":
        return engines[preference]()

    for factory in engines.values():
        try:
            return factory()
        except Exception:
            continue

    raise RuntimeError(
        "No OCR engine available. Install one via: "
        "`pip install -r requirements-ocr.txt` "
        "(paddleocr+paddlepaddle, or pytesseract with the Tesseract binary)."
    )


def _paddle_engine() -> OCREngine:
    from .paddle import PaddleOCREngine

    return PaddleOCREngine()


def _tesseract_engine() -> OCREngine:
    from .tesseract import TesseractEngine

    return TesseractEngine()
