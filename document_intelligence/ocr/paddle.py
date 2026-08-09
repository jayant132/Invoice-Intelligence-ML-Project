"""PaddleOCR engine (recommended: pure pip install, no system dependencies)."""

from pathlib import Path

from .base import OCREngine, OCRLine


class PaddleOCREngine(OCREngine):
    """Wraps PaddleOCR. Tested against paddleocr == 2.7.3."""

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR

        self._lang = lang
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract(self, image_path: str | Path) -> list[OCRLine]:
        result = self._ocr.ocr(str(image_path), cls=True)
        return [
            _to_line(box, text, score)
            for box, text, score in _iter_results(result)
            if text.strip()
        ]


def _iter_results(result):
    """Yield (box, text, score) for every recognised line.

    PaddleOCR's return shape changed across versions (2.x wraps pages in a
    nested list, 3.x flattens it). Walking the list structure keeps this
    compatible without pinning to a single layout.
    """
    if not result:
        return

    def walk(node):
        if not isinstance(node, (list, tuple)):
            return
        # A recognised line looks like: [box, (text, score)]
        if (
            len(node) == 2
            and isinstance(node[1], (list, tuple))
            and len(node[1]) == 2
            and isinstance(node[1][0], str)
        ):
            yield node[0], node[1][0], node[1][1]
            return
        for child in node:
            yield from walk(child)

    yield from walk(result)


def _to_line(box, text, score) -> OCRLine:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return OCRLine(
        text=text,
        confidence=float(score),
        bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
    )
