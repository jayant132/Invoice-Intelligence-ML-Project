"""Pydantic schemas that enforce the structure of extracted invoice data."""

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A single row from the invoice line-items table."""

    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None
    confidence: float = 1.0


class Invoice(BaseModel):
    """Structured invoice data extracted from the document."""

    invoice_number: str | None = None
    vendor: str | None = None
    customer: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    items: list[LineItem] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    low_confidence_fields: list[str] = Field(default_factory=list)
