"""Generate sample invoice images to demo the Document Intelligence pipeline.

Usage:
    python scripts/generate_sample_invoices.py

Writes three invoices to `sample_invoices/`:
  - clean_invoice.png      a tidy, digital-style invoice (valid)
  - scanned_invoice.png    a slightly rotated, grayscale scan (valid)
  - error_invoice.png      an invoice with arithmetic and date errors (invalid)
"""

from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "sample_invoices"
WIDTH, HEIGHT = 900, 1150

CLEAN = {
    "vendor": "ABC Technologies",
    "address": "Connaught Place, New Delhi, India",
    "customer": "Zenith Retail Pvt Ltd",
    "invoice_no": "INV-1023",
    "date": "2026-08-01",
    "due_date": "2026-08-31",
    "currency": "INR",
    "items": [
        {"description": "Laptop", "qty": 2, "unit_price": 50000, "amount": 100000},
        {"description": "Monitor", "qty": 2, "unit_price": 15000, "amount": 30000},
    ],
    "subtotal": 130000,
    "tax": 23400,
}

SCANNED = {
    "vendor": "Globex Supplies Inc.",
    "address": "1234 Market Street, Portland, OR, USA",
    "customer": "Northwind Traders",
    "invoice_no": "GLX-2048",
    "date": "2026-06-15",
    "due_date": "2026-07-15",
    "currency": "USD",
    "items": [
        {"description": "Widget", "qty": 10, "unit_price": 25.0, "amount": 250.0},
        {"description": "Gadget", "qty": 5, "unit_price": 40.0, "amount": 200.0},
    ],
    "subtotal": 450.0,
    "tax": 22.5,
}

ERROR = deepcopy(CLEAN)
ERROR["invoice_no"] = "INV-1024"
ERROR["date"] = "2026-08-05"
ERROR["due_date"] = "2026-08-01"  # due before invoice -> date rule fails
ERROR["items"] = [
    {"description": "Laptop", "qty": 2, "unit_price": 50000, "amount": 100000},
    # Monitor line: qty x unit price (30000) != line total (29000) -> item rule fails
    {"description": "Monitor", "qty": 2, "unit_price": 15000, "amount": 29000},
]
ERROR["subtotal"] = 129000
ERROR["tax"] = 23220
ERROR["total_override"] = 150000  # subtotal + tax = 152220 != 150000 -> totals rule fails


def _font(size: int, bold: bool = False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _money(value, currency: str) -> str:
    if currency == "INR":
        return f"\u20b9{value:,.0f}"
    return f"${value:,.2f}"


def _draw(meta: dict) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    currency = meta["currency"]

    y = 60
    draw.text((WIDTH // 2 - 90, y), "INVOICE", font=_font(40, bold=True), fill="black")
    y += 110

    draw.text((60, y), meta["vendor"], font=_font(26, bold=True), fill="black")
    y += 40
    draw.text((60, y), meta["address"], font=_font(18), fill="gray")
    y += 60

    draw.text((60, y), f"Invoice No: {meta['invoice_no']}", font=_font(20), fill="black")
    y += 34
    draw.text((60, y), f"Date: {meta['date']}", font=_font(20), fill="black")
    y += 34
    draw.text((60, y), f"Due Date: {meta['due_date']}", font=_font(20), fill="black")
    y += 44

    draw.text((60, y), f"Bill To:  {meta['customer']}", font=_font(20), fill="black")
    y += 54

    draw.line([(60, y), (WIDTH - 60, y)], fill="black", width=2)
    y += 44

    headers = ["Item", "Qty", "Unit Price", "Amount"]
    col_x = [60, 400, 560, 700]
    for header, x in zip(headers, col_x):
        draw.text((x, y), header, font=_font(20, bold=True), fill="black")
    y += 34
    draw.line([(60, y), (WIDTH - 60, y)], fill="black", width=1)
    y += 24

    for item in meta["items"]:
        draw.text((col_x[0], y), item["description"], font=_font(20), fill="black")
        draw.text((col_x[1], y), f"{item['qty']:g}", font=_font(20), fill="black")
        draw.text((col_x[2], y), _money(item["unit_price"], currency), font=_font(20), fill="black")
        draw.text((col_x[3], y), _money(item["amount"], currency), font=_font(20), fill="black")
        y += 38

    draw.line([(60, y), (WIDTH - 60, y)], fill="black", width=1)
    y += 36

    draw.text((col_x[2], y), "Subtotal:", font=_font(20), fill="black")
    draw.text((col_x[3], y), _money(meta["subtotal"], currency), font=_font(20), fill="black")
    y += 32
    draw.text((col_x[2], y), "Tax:", font=_font(20), fill="black")
    draw.text((col_x[3], y), _money(meta["tax"], currency), font=_font(20), fill="black")
    y += 36

    total = meta.get("total_override", meta["subtotal"] + meta["tax"])
    draw.text((col_x[2], y), "Total:", font=_font(24, bold=True), fill="black")
    draw.text((col_x[3], y), _money(total, currency), font=_font(24, bold=True), fill="black")

    return img


def _scanned(img: Image.Image) -> Image.Image:
    img = img.convert("L").rotate(1.8, expand=True, fillcolor=255)
    return img.filter(ImageFilter.GaussianBlur(radius=0.6))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    _draw(CLEAN).save(OUT_DIR / "clean_invoice.png")
    _scanned(_draw(SCANNED)).save(OUT_DIR / "scanned_invoice.png")
    _draw(ERROR).save(OUT_DIR / "error_invoice.png")
    print(f"Sample invoices written to {OUT_DIR}")


if __name__ == "__main__":
    main()
