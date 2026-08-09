"""Rule-based structured extraction from OCR lines with per-field confidence."""

import re
from datetime import date

from ..ocr.base import OCRLine
from .layout import DocumentLayout
from .schemas import Invoice, LineItem

CURRENCY_SYMBOLS = {"\u20b9": "INR", "$": "USD", "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "JPY"}
CURRENCY_CODES = {
    "inr": "INR",
    "usd": "USD",
    "eur": "EUR",
    "gbp": "GBP",
    "jpy": "JPY",
    "cad": "CAD",
    "aud": "AUD",
}

_LABELED_NO = re.compile(
    r"(?i)\b(?:invoice\s*(?:no\.?|number|#)?|inv\.?)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-/._]*[0-9])"
)
_STANDALONE_NO = re.compile(r"\b[A-Z]{2,6}[-/][0-9]{2,6}\b")

_DATE_PATTERNS = [
    re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"),
    re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})"),
]

_AMOUNT_RULES = [
    (
        "subtotal",
        re.compile(
            r"(?i)\bsub\s?total\b\s*[:]?\s*(?:\([^)]*\))?\s*(?:([\u20b9$€£]))?\s*([\d,]+(?:\.\d+)?)"
        ),
    ),
    (
        "tax",
        re.compile(
            r"(?i)\b(?:tax|vat|gst)\b\s*[:]?\s*(?:\([^)]*\))?\s*(?:([\u20b9$€£]))?\s*([\d,]+(?:\.\d+)?)"
        ),
    ),
    (
        "total",
        re.compile(
            r"(?i)(?:\bgrand\s+total\b|\btotal\b|\bamount\s?due\b|\bamount\s?payable\b)\s*[:]?\s*(?:\([^)]*\))?\s*(?:([\u20b9$€£]))?\s*([\d,]+(?:\.\d+)?)"
        ),
    ),
]

_VENDOR_ANCHORS = ("vendor:", "vendor name", "from:", "supplier:", "billed by:")
_CUSTOMER_ANCHORS = (
    "bill to",
    "billed to",
    "ship to",
    "shipped to",
    "invoice to",
    "sold to",
    "customer:",
)
_LABEL_RE = re.compile(
    r"(?i)(sub\s?total|tax|vat|gst|\btotal\b|amount due|discount|shipping|date|invoice\s*no)"
)


class InvoiceExtractor:
    """Extracts structured fields from OCR lines and scores their confidence."""

    def __init__(self, confidence_threshold: float = 0.65):
        self._threshold = confidence_threshold

    def extract(self, lines: list[OCRLine], layout: DocumentLayout) -> Invoice:
        invoice_number, no_conf = self._extract_invoice_number(lines)
        invoice_date, inv_conf, due_date, due_conf = self._extract_dates(lines)
        currency, cur_conf = self._extract_currency(lines)
        amounts = self._extract_amounts(lines)
        vendor, vendor_conf = self._extract_vendor(layout)
        customer, customer_conf = self._extract_customer(layout)
        items = self._extract_items(lines, layout)

        invoice = Invoice(
            invoice_number=invoice_number,
            vendor=vendor,
            customer=customer,
            invoice_date=invoice_date,
            due_date=due_date,
            currency=currency,
            subtotal=amounts.get("subtotal", (None, 0.0))[0],
            tax=amounts.get("tax", (None, 0.0))[0],
            total=amounts.get("total", (None, 0.0))[0],
            items=items,
        )

        invoice.confidence = {
            "invoice_number": no_conf,
            "vendor": vendor_conf,
            "customer": customer_conf,
            "invoice_date": inv_conf,
            "due_date": due_conf,
            "currency": cur_conf,
            "subtotal": amounts.get("subtotal", (None, 0.0))[1],
            "tax": amounts.get("tax", (None, 0.0))[1],
            "total": amounts.get("total", (None, 0.0))[1],
        }
        invoice.low_confidence_fields = sorted(
            name for name, conf in invoice.confidence.items() if conf and conf < self._threshold
        )
        return invoice

    def _extract_invoice_number(self, lines: list[OCRLine]) -> tuple[str | None, float]:
        for line in lines:
            match = _LABELED_NO.search(line.text)
            if match:
                return match.group(1), 0.9
        for line in lines:
            match = _STANDALONE_NO.search(line.text)
            if match:
                return match.group(0), 0.75
        return None, 0.0

    def _extract_dates(self, lines: list[OCRLine]) -> tuple[str | None, float, str | None, float]:
        labeled_invoice, labeled_due, unlabeled = [], [], []
        for line in lines:
            text = line.text.lower()
            for pattern in _DATE_PATTERNS:
                for match in pattern.finditer(line.text):
                    iso = _normalize_date(*match.groups())
                    if not iso:
                        continue
                    if "due" in text:
                        labeled_due.append(iso)
                    elif "invoice" in text or "date" in text:
                        labeled_invoice.append(iso)
                    else:
                        unlabeled.append(iso)

        invoice_date = labeled_invoice[0] if labeled_invoice else (unlabeled[0] if unlabeled else None)
        due_date = labeled_due[0] if labeled_due else None
        if due_date is None and invoice_date is not None and len(unlabeled) >= 2:
            due_date = unlabeled[1]

        invoice_conf = 0.9 if labeled_invoice else (0.6 if invoice_date else 0.0)
        due_conf = 0.9 if labeled_due else (0.6 if due_date else 0.0)
        return invoice_date, invoice_conf, due_date, due_conf

    def _extract_currency(self, lines: list[OCRLine]) -> tuple[str | None, float]:
        for line in lines:
            for symbol, code in CURRENCY_SYMBOLS.items():
                if symbol in line.text:
                    return code, 0.95
        for line in lines:
            for match in re.finditer(r"\b([a-z]{3})\b", line.text, re.IGNORECASE):
                code = match.group(1).lower()
                if code in CURRENCY_CODES:
                    return CURRENCY_CODES[code], 0.9
        return None, 0.0

    def _extract_amounts(self, lines: list[OCRLine]) -> dict[str, tuple[float, float]]:
        amounts: dict[str, tuple[float, float]] = {}
        for key, pattern in _AMOUNT_RULES:
            for line in lines:
                match = pattern.search(line.text)
                if match:
                    symbol, number = match.group(1), match.group(2)
                    amounts[key] = (
                        float(number.replace(",", "")),
                        0.9 if symbol else 0.8,
                    )
        return amounts

    def _extract_vendor(self, layout: DocumentLayout) -> tuple[str | None, float]:
        for line in layout.vendor_lines:
            if _starts_with_any(line.text.lower(), _VENDOR_ANCHORS):
                return _strip_label(line.text), 0.9
        if layout.vendor_lines:
            return layout.vendor_lines[0].text, 0.7
        return None, 0.0

    def _extract_customer(self, layout: DocumentLayout) -> tuple[str | None, float]:
        for line in layout.customer_lines:
            if _starts_with_any(line.text.lower(), _CUSTOMER_ANCHORS):
                return _strip_label(line.text), 0.7
        if layout.customer_lines:
            return layout.customer_lines[0].text, 0.6
        return None, 0.0

    def _extract_items(self, lines: list[OCRLine], layout: DocumentLayout) -> list[LineItem]:
        candidates = [line for line in layout.item_lines if _count_numbers(line.text) >= 2]
        if not candidates:
            candidates = [
                line
                for line in lines
                if _count_numbers(line.text) >= 2 and not _LABEL_RE.search(line.text)
            ]

        items: list[LineItem] = []
        for line in candidates:
            item = _parse_item_row(line.text)
            if item:
                items.append(item)
        return items


def _parse_item_row(text: str) -> LineItem | None:
    values: list[float] = []
    description: list[str] = []
    for token in text.split():
        number = _parse_number(token)
        if number is not None:
            values.append(number)
        elif token in CURRENCY_SYMBOLS:
            continue
        else:
            description.append(token)

    if not values or not description:
        return None

    if len(values) >= 3:
        quantity, unit_price, amount = values[0], values[1], values[2]
    elif len(values) == 2:
        quantity, amount = values[0], values[1]
        unit_price = amount / quantity if quantity else None
    else:
        quantity, unit_price, amount = None, None, values[0]

    consistent = (
        quantity is not None
        and unit_price is not None
        and amount is not None
        and abs(quantity * unit_price - amount) < 1.0
    )
    return LineItem(
        description=" ".join(description),
        quantity=quantity,
        unit_price=unit_price,
        amount=amount,
        confidence=round(0.85 if consistent else 0.7, 2),
    )


def _count_numbers(text: str) -> int:
    return sum(1 for token in text.split() if _parse_number(token) is not None)


def _parse_number(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("%", "")
    for symbol in CURRENCY_SYMBOLS:
        cleaned = cleaned.replace(symbol, "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_date(*parts: str) -> str | None:
    a, b, c = (int(p) for p in parts)
    if a > 31:  # year-month-day (4-digit year first)
        year, month, day = a, b, c
    else:
        year = c + (2000 if c < 100 else 0)
        if a > 12:
            day, month = a, b
        elif b > 12:
            month, day = a, b
        else:
            day, month = a, b  # ambiguous, default to day-first
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _strip_label(text: str) -> str:
    text = text.strip().strip(":;,.")
    match = re.match(
        r"(?i)^(?:vendor|from|supplier|billed by|bill to|billed to|ship to|shipped to|"
        r"invoice to|sold to|customer)\s*[:.]?\s*(.*)$",
        text,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text


def _starts_with_any(text: str, anchors: tuple[str, ...]) -> bool:
    return any(text.startswith(anchor) for anchor in anchors)
