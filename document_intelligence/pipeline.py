"""End-to-end invoice document pipeline: OCR -> layout -> extraction -> validation."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .extraction.extractor import InvoiceExtractor
from .extraction.layout import DocumentLayout, analyze_layout
from .extraction.schemas import Invoice
from .ocr.base import OCREngine, OCRLine
from .ocr.factory import get_ocr_engine
from .validation.rules import ValidationReport, validate_invoice


@dataclass
class ProcessingResult:
    """Everything produced by processing a single invoice document."""

    source_path: Path
    ocr_lines: list[OCRLine]
    layout: DocumentLayout
    invoice: Invoice
    validation: ValidationReport
    needs_human_review: bool

    def to_dict(self) -> dict:
        return {
            "source": str(self.source_path),
            "invoice": self.invoice.model_dump(),
            "validation": self.validation.to_dict(),
            "needs_human_review": self.needs_human_review,
            "layout": self.layout.as_dict(),
        }


def process_invoice(
    source_path: str | Path,
    engine: str | OCREngine = "auto",
    extractor: InvoiceExtractor | None = None,
) -> ProcessingResult:
    """Run the full document intelligence pipeline on one invoice file."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Invoice file not found: {source_path}")

    image_path = _render_pdf(source_path) if source_path.suffix.lower() == ".pdf" else source_path
    ocr_engine = engine if isinstance(engine, OCREngine) else get_ocr_engine(engine)

    lines = ocr_engine.extract(image_path)
    layout = analyze_layout(lines)
    invoice = (extractor or InvoiceExtractor()).extract(lines, layout)
    report = validate_invoice(invoice)

    return ProcessingResult(
        source_path=source_path,
        ocr_lines=lines,
        layout=layout,
        invoice=invoice,
        validation=report,
        needs_human_review=report.needs_human_review or bool(invoice.low_confidence_fields),
    )


def _render_pdf(pdf_path: Path) -> Path:
    """Render the first PDF page to a temporary PNG image."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    bitmap = pdf[0].render(scale=2.0)
    image = bitmap.to_pil()

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    image.save(tmp_path)
    return Path(tmp_path)
