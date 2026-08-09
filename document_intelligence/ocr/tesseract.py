"""Tesseract engine (requires the Tesseract binary installed on the system)."""

from collections import defaultdict
from pathlib import Path
from statistics import mean

from .base import OCREngine, OCRLine


class TesseractEngine(OCREngine):
    def __init__(self, lang: str = "eng"):
        import pytesseract

        pytesseract.get_tesseract_version()  # raises if the binary is missing
        self._pytesseract = pytesseract
        self._lang = lang

    def extract(self, image_path: str | Path) -> list[OCRLine]:
        from PIL import Image

        data = self._pytesseract.image_to_data(
            Image.open(image_path),
            lang=self._lang,
            output_type=self._pytesseract.Output.DICT,
        )

        # Group words back into lines using Tesseract's block/paragraph/line ids.
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i in range(len(data["text"])):
            if not data["text"][i].strip():
                continue
            groups[
                (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            ].append(i)

        lines: list[OCRLine] = []
        for indices in groups.values():
            indices = sorted(indices, key=lambda i: data["left"][i])
            text = " ".join(data["text"][i].strip() for i in indices)

            confs = [float(data["conf"][i]) for i in indices if data["conf"][i] != "-1"]
            confidence = mean(confs) / 100.0 if confs else 0.0

            left = min(data["left"][i] for i in indices)
            top = min(data["top"][i] for i in indices)
            right = max(data["left"][i] + data["width"][i] for i in indices)
            bottom = max(data["top"][i] + data["height"][i] for i in indices)

            lines.append(
                OCRLine(
                    text=text,
                    confidence=round(confidence, 3),
                    bbox=(left, top, right, bottom),
                )
            )
        return lines
