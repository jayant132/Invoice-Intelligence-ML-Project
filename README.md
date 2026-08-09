<div align="center">

# 🧾 Invoice Intelligence

### Document AI · freight cost prediction · invoice fraud detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Schemas-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-OCR-3B5BA5)](https://github.com/PaddlePaddle/PaddleOCR)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end invoice intelligence system built around a **Document AI pipeline** — turn any
invoice image or PDF into **structured, validated data** — plus two classic ML pipelines for
freight cost prediction and fraud/risk flagging.

</div>

---

## 📌 Overview

Invoices arrive as messy PDFs, scans and photos. This project tackles the three problems every
procurement team faces when processing them:

1. **Understand the document** — OCR the invoice, work out its layout (header, vendor, customer,
   line items, totals) and extract a structured, schema-validated representation with per-field
   confidence scores.
2. **How much should freight cost?** — regression to predict expected freight, so inflated
   charges stand out.
3. **Which invoices need a second look?** — classification to flag invoices whose amounts or
   delivery timing look wrong, plus a **validation engine** that catches arithmetic and date
   errors automatically.

```
                    ┌────────────────────────────────────────────────┐
                    │          DOCUMENT INTELLIGENCE PIPELINE         │
                    │                                                │
 Invoice image/PDF  │  1. OCR ──► 2. Layout ──► 3. Extraction ──►    │
   ───────────────► │       raw text      regions      Pydantic      │
                    │        + coords     header      schema +       │
                    │                     vendor      confidence     │
                    │                     items                      │
                    │                     totals                     │
                    │       4. Validation engine                     │
                    │          subtotal+tax=total?                   │
                    │          qty×price=line total?                 │
                    │          invoice date ≤ due date?              │
                    │          currency valid? required fields?      │
                    │                    │                           │
                    │          ┌─────────┴─────────┐                 │
                    │          │                  │                 │
                    │       VALID             NEEDS REVIEW           │
                    │          │                  │                 │
                    └──────────┼──────────────────┼─────────────────┘
                               ▼                  ▼
                          JSON output       human review
                          (CLI / UI)        (confidence < 65%)

  ┌───────────────────────┐        ┌───────────────────────┐
  │  freight_cost_prediction│        │      invoice_flagging  │
  │  Regression: predict    │        │  Classification: flag  │
  │  Freight_Cost from       │        │  risky invoices from   │
  │  invoice Dollars         │        │  dollar gaps & delays  │
  └───────────┬───────────┘        └───────────┬───────────┘
              └───────────────┬────────────────┘
                              ▼
                      Streamlit app (app.py)
```

---

## 🧠 Document Intelligence — the core upgrade

A 5-stage pipeline that moves the project from "read text" to **"understand documents"**:

### 1. OCR

[`document_intelligence/ocr/`](document_intelligence/ocr/) turns an image (or PDF, rendered via
`pypdfium2`) into text lines with coordinates. The engine is pluggable behind a small strategy
interface:

- **PaddleOCR** (recommended) — pure pip install, strong on clean and scanned documents.
- **Tesseract** — fallback if the Tesseract binary is installed.

The factory auto-detects whichever is available, so no code changes are needed to switch.

### 2. Document layout understanding

[`document_intelligence/extraction/layout.py`](document_intelligence/extraction/layout.py)
classifies every OCR line into a semantic region using keyword anchors:

```
header          INVOICE
vendor          ABC Technologies · Delhi, India
metadata        Invoice No: INV-1023 · Date: 2026-08-01
customer        Bill To: Zenith Retail Pvt Ltd
column_header   Item  Qty  Unit Price  Amount
items           Laptop 2 ₹50,000 ₹100,000
totals          Subtotal ₹130,000 · Tax ₹23,400 · Total ₹153,400
```

### 3. Structured extraction (Pydantic schemas)

[`document_intelligence/extraction/schemas.py`](document_intelligence/extraction/schemas.py)
enforces the output shape with Pydantic, so extraction always returns a valid `Invoice` model —
**structured outputs + validation instead of free-form text**. Example output:

```json
{
  "invoice_number": "INV-1023",
  "vendor": "ABC Technologies",
  "customer": "Zenith Retail Pvt Ltd",
  "invoice_date": "2026-08-01",
  "due_date": "2026-08-31",
  "currency": "INR",
  "subtotal": 130000,
  "tax": 23400,
  "total": 153400,
  "items": [
    {"description": "Laptop", "quantity": 2, "unit_price": 50000, "amount": 100000},
    {"description": "Monitor", "quantity": 2, "unit_price": 15000, "amount": 30000}
  ]
}
```

> The extractor is rule/regex based so the project runs **offline with no API keys**.
> Swapping in an LLM that emits a `LineItem`/`Invoice` schema (e.g. `instructor`) is a
> drop-in replacement — the Pydantic models and confidence plumbing stay the same.

### 4. Confidence scores

Every field carries a confidence in `[0, 1]` derived from extraction quality (labelled match vs.
heuristic, currency symbol present, etc.). Fields below **65%** are listed in
`low_confidence_fields`, which triggers a **"human verification recommended"** banner — a basic
human-in-the-loop AI workflow:

```
Total: ₹153,400          Vendor: ABC Technolog...    ⚠ Human verification
Confidence: 98%          Confidence: 61%              recommended
```

### 5. Validation engine

[`document_intelligence/validation/rules.py`](document_intelligence/validation/rules.py) checks
each invoice against business rules and reports `PASS` / `FAIL` / `WARN`:

| Rule | Checks |
|---|---|
| `required_fields` | invoice number, date, vendor and total exist |
| `currency` | currency is a known ISO code |
| `totals` | `subtotal + tax = total` |
| `line_items` | `qty × unit_price = line total` and `Σ line totals = subtotal` |
| `dates` | invoice date ≤ due date |

A single `FAIL` routes the invoice to **human review** instead of the database.

---

## 🤖 The ML pipelines

### 1. Freight Cost Prediction

Predicts `Freight_Cost` from the invoice amount (`Dollars`) using regression, so unusually high
freight charges stand out against what's expected.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 7.81 | 10.53 | 96.86% |
| **Decision Tree (depth=5)** | **7.32** | **10.10** | **97.12%** |
| Random Forest (depth=3) | 9.00 | 11.85 | 96.03% |

### 2. Invoice Fraud/Risk Flagging

Classifies each invoice as **normal (0)** or **flagged (1)** based on two real risk signals:

- The invoice total doesn't match the purchased item total by more than **$350**
- The average delay between PO date and receiving date exceeds **10 days**

The flagging label is a *relationship between columns*, which tree models can't represent
natively — so it's engineered explicitly:

```python
df["Dollar_Gap"] = (df["Invoice_Dollars"] - df["Total_Item_Dollars"]).abs()
```

This single feature took recall on flagged invoices from **17% → 99%**.

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Normal (0) | 1.00 | 1.00 | 1.00 |
| **Flagged (1)** | **1.00** | **0.99** | **0.99** |

*4,878 invoices · 14% flagged · stratified 80/20 split · tuned Random Forest*

---

## 📁 Project Structure

```
Invoice-Intelligence-ML-Project/
├── app.py                              # Streamlit UI (3 tabs)
├── requirements.txt                    # Core dependencies
├── requirements-ocr.txt                # Optional OCR engines
├── data/
│   └── inventory.db                    # SQLite source database
├── document_intelligence/              # Document AI pipeline
│   ├── __init__.py
│   ├── pipeline.py                     # OCR → layout → extraction → validation
│   ├── cli.py                          # python -m document_intelligence.cli
│   ├── streamlit_view.py               # Streamlit tab UI
│   ├── ocr/
│   │   ├── base.py                     # OCREngine interface + OCRLine
│   │   ├── paddle.py                   # PaddleOCR engine
│   │   ├── tesseract.py                # Tesseract engine
│   │   └── factory.py                  # engine auto-detection
│   ├── extraction/
│   │   ├── schemas.py                  # Pydantic Invoice / LineItem models
│   │   ├── layout.py                   # region classification
│   │   └── extractor.py                # rule-based field extraction + confidence
│   └── validation/
│       └── rules.py                    # validation engine
├── scripts/
│   └── generate_sample_invoices.py     # creates demo invoices
├── sample_invoices/                    # generated demo invoices (run the script)
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── train.py
│   └── models/
│       └── predict_freight_model.pkl
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   ├── train.py
│   └── models/
│       ├── random_forest.pkl
│       └── scaler.pkl
├── inferencing/
│   ├── freight_cost.py
│   └── predict_invoice_flag.py
├── notebooks/
│   └── Predicting Freight Cost.ipynb
├── LICENSE
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/Invoice-Intelligence-ML-Project.git
cd Invoice-Intelligence-ML-Project

pip install -r requirements.txt          # core app
pip install -r requirements-ocr.txt      # OCR engines (optional but recommended)
```

> **PaddleOCR** downloads its recognition models on first run — the first extraction takes a
> minute, later ones are fast. Prefer **Tesseract**? Install the binary on your system and
> `pip install pytesseract`, then pick `tesseract` in the UI or `--engine tesseract` in the CLI.

---

## 🚀 Usage

### Generate demo invoices

```bash
python scripts/generate_sample_invoices.py
```

Writes three invoices to `sample_invoices/`: a clean one, a rotated grayscale "scan", and one
full of deliberate arithmetic + date errors.

### Run the Document Intelligence CLI

```bash
python -m document_intelligence.cli sample_invoices/clean_invoice.png --pretty
python -m document_intelligence.cli sample_invoices/error_invoice.png --pretty
python -m document_intelligence.cli my_invoice.pdf --engine tesseract
```

Output is JSON: the structured invoice, per-field confidence, layout regions and the
validation report with `needs_human_review`.

### Streamlit app

```bash
streamlit run app.py
```

Three tabs:

- **🧠 Document Intelligence** — upload an image/PDF or pick a sample; see the structured JSON,
  confidence bars, layout regions and validation report.
- **🚩 Invoice Flagging** — enter invoice/PO details to check risk.
- **📦 Freight Cost Prediction** — predict expected freight from the invoice amount.

### Train the ML models

```bash
cd invoice_flagging && python train.py
cd ../freight_cost_prediction && python train.py
```

Each run compares several algorithms and saves the best model to its `models/` folder.

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Pydantic` · `PaddleOCR` / `Tesseract` · `pandas` · `scikit-learn` ·
`SQLite` · `joblib` · `pypdfium2`

---

## 🔮 Future Improvements

- [ ] LLM-backed extraction (`instructor` + `gpt/llama`) that emits the existing `Invoice` schema
      for harder documents, with rule-based extractor kept as the offline fallback
- [ ] Table cell coords → bounding-box-aware line-item parsing for multi-column layouts
- [ ] SHAP explainability on the flagging model so flags come with a human-readable reason
- [ ] Automated retraining schedule as new invoice data arrives

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

<div align="center">

Built as a hands-on exploration of Document AI and end-to-end ML pipelines — from raw
invoices and SQL to validated, structured data.

</div>
