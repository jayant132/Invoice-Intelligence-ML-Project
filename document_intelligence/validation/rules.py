"""Business rules that validate an extracted invoice.

Each rule returns PASS, FAIL or WARN. A FAIL means the invoice must be
reviewed by a human before it is trusted; WARN means the rule could not be
fully evaluated because data is missing.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from ..extraction.schemas import Invoice

VALID_CURRENCIES = {"INR", "USD", "EUR", "GBP", "JPY", "CAD", "AUD"}
REQUIRED_FIELDS = ("invoice_number", "invoice_date", "vendor", "total")
AMOUNT_TOLERANCE = 1.0

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_WARN = "WARN"
STATUS_SKIP = "SKIP"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation rule."""

    rule: str
    status: str
    message: str


@dataclass
class ValidationReport:
    """Collection of rule outcomes with an aggregate status."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = {result.status for result in self.results}
        if STATUS_FAIL in statuses:
            return STATUS_FAIL
        if STATUS_WARN in statuses:
            return STATUS_WARN
        return STATUS_PASS

    @property
    def needs_human_review(self) -> bool:
        return self.status == STATUS_FAIL

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "needs_human_review": self.needs_human_review,
            "results": [
                {"rule": r.rule, "status": r.status, "message": r.message}
                for r in self.results
            ],
        }


def validate_invoice(invoice: Invoice) -> ValidationReport:
    """Run every rule against the extracted invoice."""
    report = ValidationReport()
    report.results.extend(
        [
            _check_required_fields(invoice),
            _check_currency(invoice),
            _check_totals(invoice),
            _check_line_items(invoice),
            _check_dates(invoice),
        ]
    )
    return report


def _check_required_fields(invoice: Invoice) -> ValidationResult:
    missing = [name for name in REQUIRED_FIELDS if getattr(invoice, name) is None]
    if missing:
        return ValidationResult(
            "required_fields",
            STATUS_FAIL,
            f"Missing required fields: {', '.join(missing)}.",
        )
    return ValidationResult("required_fields", STATUS_PASS, "All required fields present.")


def _check_currency(invoice: Invoice) -> ValidationResult:
    if invoice.currency is None:
        return ValidationResult("currency", STATUS_WARN, "Currency not detected.")
    if invoice.currency.upper() not in VALID_CURRENCIES:
        return ValidationResult(
            "currency", STATUS_FAIL, f"Unsupported currency: {invoice.currency}."
        )
    return ValidationResult(
        "currency", STATUS_PASS, f"Currency {invoice.currency} is valid."
    )


def _check_totals(invoice: Invoice) -> ValidationResult:
    if invoice.subtotal is None and invoice.tax is None:
        return ValidationResult(
            "totals", STATUS_WARN, "Subtotal and tax missing; cannot verify total."
        )
    if invoice.total is None:
        return ValidationResult(
            "totals", STATUS_WARN, "Total missing; cannot verify arithmetic."
        )
    expected = (invoice.subtotal or 0.0) + (invoice.tax or 0.0)
    if abs(expected - invoice.total) <= AMOUNT_TOLERANCE:
        return ValidationResult(
            "totals", STATUS_PASS, f"Subtotal + tax = total ({invoice.total:,.2f})."
        )
    return ValidationResult(
        "totals",
        STATUS_FAIL,
        f"Subtotal + tax ({expected:,.2f}) does not match total ({invoice.total:,.2f}).",
    )


def _check_line_items(invoice: Invoice) -> ValidationResult:
    if not invoice.items:
        return ValidationResult(
            "line_items", STATUS_WARN, "No line items extracted."
        )

    errors: list[str] = []
    incomplete = 0
    line_total = 0.0
    for item in invoice.items:
        if item.amount is None:
            incomplete += 1
            continue
        line_total += item.amount
        if item.quantity is not None and item.unit_price is not None:
            expected = item.quantity * item.unit_price
            if abs(expected - item.amount) > AMOUNT_TOLERANCE:
                errors.append(
                    f"{item.description!r}: qty {item.quantity:g} x price "
                    f"{item.unit_price:g} = {expected:g}, but the line total is "
                    f"{item.amount:g}."
                )

    if invoice.subtotal is not None and line_total and abs(line_total - invoice.subtotal) > AMOUNT_TOLERANCE:
        errors.append(
            f"Sum of line totals ({line_total:g}) does not match subtotal "
            f"({invoice.subtotal:g})."
        )

    if errors:
        return ValidationResult("line_items", STATUS_FAIL, " ".join(errors))
    if incomplete:
        return ValidationResult(
            "line_items",
            STATUS_WARN,
            f"{incomplete} item(s) missing amount data.",
        )
    return ValidationResult(
        "line_items", STATUS_PASS, "Line-item arithmetic is consistent."
    )


def _check_dates(invoice: Invoice) -> ValidationResult:
    if invoice.invoice_date is None:
        return ValidationResult("dates", STATUS_WARN, "Invoice date missing.")
    if invoice.due_date is None:
        return ValidationResult("dates", STATUS_SKIP, "No due date to compare.")

    invoice_date = _parse_date(invoice.invoice_date)
    due_date = _parse_date(invoice.due_date)
    if invoice_date is None or due_date is None:
        return ValidationResult(
            "dates", STATUS_WARN, "Invoice or due date could not be parsed."
        )
    if invoice_date <= due_date:
        return ValidationResult(
            "dates",
            STATUS_PASS,
            f"Invoice date ({invoice_date}) is on or before due date ({due_date}).",
        )
    return ValidationResult(
        "dates",
        STATUS_FAIL,
        f"Invoice date ({invoice_date}) is after due date ({due_date}).",
    )


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        pass
    match = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$", value.strip())
    if not match:
        return None
    a, b, c = (int(part) for part in match.groups())
    year = c if c > 99 else c + 2000
    if a > 12:
        day, month = a, b
    elif b > 12:
        month, day = a, b
    else:
        day, month = a, b  # ambiguous, default to day-first
    try:
        return date(year, month, day)
    except ValueError:
        return None
