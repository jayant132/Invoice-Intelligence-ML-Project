"""Rule-based document layout understanding.

Assigns each OCR line to a semantic region of the invoice so downstream
extraction knows whether text is part of the header, the vendor/customer
blocks, the line-items table or the totals section.
"""

import re
from dataclasses import dataclass

from ..ocr.base import OCRLine

_TITLE_RE = re.compile(r"^invoice\b", re.IGNORECASE)
_METADATA_ANCHORS = (
    "invoice no",
    "invoice number",
    "invoice #",
    "invoice date",
    "due date",
    "po number",
    "po no",
    "reference",
    "payment terms",
    "date:",
)
_CUSTOMER_ANCHORS = (
    "bill to",
    "billed to",
    "ship to",
    "shipped to",
    "sold to",
    "invoice to",
    "customer:",
    "customer name",
)
_COLUMN_ANCHORS = (
    "qty",
    "quantity",
    "description",
    "unit price",
    "unit rate",
    "rate",
    "amount",
    "item",
    "price",
    "particulars",
)
_TOTALS_ANCHORS = (
    "subtotal",
    "grand total",
    "amount due",
    "amount payable",
    "balance due",
    "discount",
    "shipping",
    "freight",
    "tax",
    "vat",
    "gst",
    "total",
)


@dataclass
class DocumentLayout:
    """Semantic regions of the invoice, in reading order."""

    header_lines: list[OCRLine]
    vendor_lines: list[OCRLine]
    customer_lines: list[OCRLine]
    column_header_lines: list[OCRLine]
    item_lines: list[OCRLine]
    totals_lines: list[OCRLine]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            name: [line.text for line in lines]
            for name, lines in (
                ("header", self.header_lines),
                ("vendor", self.vendor_lines),
                ("customer", self.customer_lines),
                ("column_header", self.column_header_lines),
                ("items", self.item_lines),
                ("totals", self.totals_lines),
            )
        }


def analyze_layout(lines: list[OCRLine]) -> DocumentLayout:
    """Classify OCR lines into invoice regions using keyword anchors."""
    sections = {
        "header_lines": [],
        "vendor_lines": [],
        "customer_lines": [],
        "column_header_lines": [],
        "item_lines": [],
        "totals_lines": [],
    }

    phase = "header"  # header -> vendor -> customer -> items -> totals
    for line in lines:
        text = line.text.strip().lower()

        if _is_separator(text):
            continue

        if phase == "header":
            if _is_title(text) or _is_metadata(text) or len(sections["header_lines"]) >= 5:
                sections["header_lines"].append(line)
                phase = "vendor"
            else:
                sections["header_lines"].append(line)

        elif phase == "vendor":
            if _starts_with_any(text, _CUSTOMER_ANCHORS):
                sections["customer_lines"].append(line)
                phase = "customer"
            elif _starts_with_any(text, _TOTALS_ANCHORS):
                sections["totals_lines"].append(line)
                phase = "totals"
            elif _is_metadata(text):
                sections["header_lines"].append(line)
            else:
                sections["vendor_lines"].append(line)

        elif phase == "customer":
            if _starts_with_any(text, _COLUMN_ANCHORS):
                sections["column_header_lines"].append(line)
                phase = "items"
            elif _starts_with_any(text, _TOTALS_ANCHORS):
                sections["totals_lines"].append(line)
                phase = "totals"
            else:
                sections["customer_lines"].append(line)

        elif phase == "items":
            if _starts_with_any(text, _TOTALS_ANCHORS):
                sections["totals_lines"].append(line)
                phase = "totals"
            else:
                sections["item_lines"].append(line)

        else:  # totals
            sections["totals_lines"].append(line)

    return DocumentLayout(**sections)


def _is_separator(text: str) -> bool:
    return bool(text) and bool(re.fullmatch(r"[\s\-_=~.*]+", text))


def _is_title(text: str) -> bool:
    return bool(_TITLE_RE.match(text)) and not _is_metadata(text)


def _is_metadata(text: str) -> bool:
    return _starts_with_any(text, _METADATA_ANCHORS)


def _starts_with_any(text: str, anchors: tuple[str, ...]) -> bool:
    return any(text.startswith(anchor) for anchor in anchors)
