"""Command-line entry point for the invoice document pipeline.

Usage:
    python -m document_intelligence.cli path/to/invoice.png --pretty
    python -m document_intelligence.cli path/to/invoice.pdf --engine tesseract
"""

import argparse
import json
import sys
from pathlib import Path

from .pipeline import process_invoice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the invoice document pipeline (OCR -> extraction -> validation)."
    )
    parser.add_argument("source", type=Path, help="Invoice image (PNG/JPG) or PDF.")
    parser.add_argument(
        "--engine",
        choices=["auto", "paddle", "tesseract"],
        default="auto",
        help="OCR engine to use (default: auto-detect).",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the JSON output."
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        parser.error(f"file not found: {args.source}")

    result = process_invoice(args.source, engine=args.engine)
    json.dump(result.to_dict(), sys.stdout, indent=2 if args.pretty else None)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
