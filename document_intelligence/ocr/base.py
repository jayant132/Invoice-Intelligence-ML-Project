"""OCR engine interface and the line model shared across the pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OCRLine:
    """A single line of text recognised by an OCR engine.

    Attributes:
        text: The recognised text.
        confidence: Engine confidence in the range [0, 1].
        bbox: Axis-aligned bounding box as (x1, y1, x2, y2) in pixels.
    """

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def y_center(self) -> float:
        """Vertical centre of the line, used for layout ordering."""
        return (self.bbox[1] + self.bbox[3]) / 2


class OCREngine(ABC):
    """Strategy interface implemented by concrete OCR engines."""

    @abstractmethod
    def extract(self, image_path: str | Path) -> list[OCRLine]:
        """Run OCR on an image and return the recognised lines in reading order."""
