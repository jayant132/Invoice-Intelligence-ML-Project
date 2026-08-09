"""Streamlit tab that renders the Document Intelligence pipeline in the app."""

import os
import tempfile
from pathlib import Path

import streamlit as st

from .pipeline import process_invoice
from .validation.rules import ValidationReport

STATUS_EMOJI = {"PASS": "\u2705", "FAIL": "\u274c", "WARN": "\u26a0\ufe0f", "SKIP": "\u23ed\ufe0f"}

FIELD_LABELS = [
    ("invoice_number", "Invoice Number"),
    ("vendor", "Vendor"),
    ("customer", "Customer"),
    ("invoice_date", "Invoice Date"),
    ("due_date", "Due Date"),
    ("currency", "Currency"),
    ("subtotal", "Subtotal"),
    ("tax", "Tax"),
    ("total", "Total"),
]

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_invoices"


@st.cache_resource
def _get_engine(engine_name: str):
    from .ocr.factory import get_ocr_engine

    return get_ocr_engine(engine_name)


def render_document_intelligence_tab() -> None:
    st.caption(
        "OCR \u2192 layout understanding \u2192 structured extraction (Pydantic) \u2192 "
        "confidence scores \u2192 validation engine"
    )

    col1, col2 = st.columns(2)
    engine_name = col1.selectbox(
        "OCR engine",
        ["auto", "paddle", "tesseract"],
        help="auto tries PaddleOCR first, then Tesseract.",
    )
    source = _choose_source(col2)

    if st.button("Extract Invoice", type="primary", use_container_width=True):
        if source is None:
            st.warning("Upload an invoice image/PDF or generate sample invoices first.")
            return
        try:
            with st.spinner("Running OCR, extraction and validation\u2026"):
                result = process_invoice(source, engine=_get_engine(engine_name))
            _render_results(result)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.info(
                "Tip: run `pip install -r requirements-ocr.txt` to enable PaddleOCR, "
                "or install the Tesseract binary for the pytesseract fallback."
            )


def _choose_source(column) -> Path | None:
    uploaded = column.file_uploader(
        "Upload an invoice", type=["png", "jpg", "jpeg", "pdf"]
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            handle.write(uploaded.getvalue())
        return Path(path)

    samples = sorted(SAMPLE_DIR.glob("*")) if SAMPLE_DIR.exists() else []
    if samples:
        name = column.selectbox(
            "\u2026or pick a generated sample", [sample.name for sample in samples]
        )
        return SAMPLE_DIR / name

    column.caption(
        "No `sample_invoices/` folder yet. Generate samples with "
        "`python scripts/generate_sample_invoices.py`."
    )
    return None


def _render_results(result) -> None:
    if result.needs_human_review:
        st.error(
            "\u26a0\ufe0f Human verification recommended \u2014 see the "
            "validation and confidence details below."
        )

    status = result.validation.status
    st.metric("Validation status", status)
    st.metric("Line items", len(result.invoice.items))

    tab_data, tab_validation, tab_ocr = st.tabs(
        ["Structured data", "Validation", "OCR & layout"]
    )
    with tab_data:
        _render_structured_data(result)
    with tab_validation:
        _render_validation(result.validation)
    with tab_ocr:
        _render_ocr(result)


def _render_structured_data(result) -> None:
    if result.invoice.low_confidence_fields:
        st.warning(
            "Low-confidence fields (human verification recommended): "
            + ", ".join(result.invoice.low_confidence_fields)
        )
    st.json(result.invoice.model_dump())

    st.subheader("Field confidence")
    for field, label in FIELD_LABELS:
        value = getattr(result.invoice, field)
        if value is None:
            continue
        _confidence_row(label, value, result.invoice.confidence.get(field, 0.0))

    if result.invoice.items:
        st.subheader("Line item confidence")
        for index, item in enumerate(result.invoice.items, start=1):
            _confidence_row(f"Item {index}: {item.description}", item.amount, item.confidence)


def _confidence_row(label, value, confidence: float) -> None:
    pct = round(confidence * 100)
    cols = st.columns([3, 3, 4])
    cols[0].markdown(f"**{label}**")
    cols[1].write(value)
    cols[2].markdown(f"{'█' * (pct // 10)}{'░' * (10 - pct // 10)}  **{pct}%**")


def _render_validation(report: ValidationReport) -> None:
    for result in report.results:
        emoji = STATUS_EMOJI.get(result.status, "\u2022")
        message = f"{emoji} **{result.rule}** \u2014 {result.message}"
        if result.status == "PASS":
            st.success(message)
        elif result.status == "FAIL":
            st.error(message)
        elif result.status == "WARN":
            st.warning(message)
        else:
            st.caption(message)


def _render_ocr(result) -> None:
    with st.expander("Raw OCR text"):
        st.code("\n".join(line.text for line in result.ocr_lines), language="text")

    st.subheader("Layout regions")
    for region, texts in result.layout.as_dict().items():
        if texts:
            st.markdown(f"**{region.replace('_', ' ').title()}** ({len(texts)})")
            st.write(texts)
