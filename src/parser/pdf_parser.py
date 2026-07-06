"""PDF resume parser.

Primary engine: PyMuPDF (`fitz`) — fast, handles most layouts including
simple two-column resumes reasonably well because it extracts text in
block/line order rather than raw content-stream order.

Fallback engine: `pdfplumber` — used only if PyMuPDF raises or returns
nothing, since the two libraries fail on different edge cases (pdfplumber
is sturdier on some malformed cross-reference tables; PyMuPDF is sturdier
on some encrypted-but-empty-password files).

Design choice: we deliberately do NOT attempt OCR here (no pytesseract /
easyocr dependency). Image-only ("scanned") PDFs are detected and reported
as a clear, typed error instead of silently returning nothing — see the
README "Limitations" section for the trade-off.
"""

from __future__ import annotations

from src.utils.exceptions import (
    CorruptedDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def _extract_with_pymupdf(file_path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise EncryptedDocumentError(f"PDF is password-protected: {file_path}")

        pages_text = []
        for page in doc:
            # "blocks" sort keeps multi-column resumes closer to reading order
            # than the raw stream order that plain page.get_text() would give.
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))  # (row band, x)
            pages_text.append("\n".join(b[4] for b in blocks if b[4].strip()))
        return "\n\n".join(pages_text)
    finally:
        doc.close()


def _extract_with_pdfplumber(file_path: str) -> str:
    import pdfplumber

    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        if getattr(pdf, "is_encrypted", False):
            raise EncryptedDocumentError(f"PDF is password-protected: {file_path}")
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    pages_text.append(" | ".join(cell or "" for cell in row))
            pages_text.append(text)
    return "\n\n".join(t for t in pages_text if t)


def parse_pdf(file_path: str) -> str:
    """Extract raw text from a PDF resume.

    Tries PyMuPDF first, falls back to pdfplumber on failure, and raises a
    typed, descriptive exception (never a bare crash) if both fail or if the
    document turns out to be encrypted / image-only / empty.
    """
    text = ""
    primary_error: Exception | None = None

    try:
        text = _extract_with_pymupdf(file_path)
    except EncryptedDocumentError:
        raise
    except Exception as exc:  # noqa: BLE001 - intentionally broad, we fall back
        primary_error = exc
        log.warning(f"PyMuPDF failed on {file_path} ({exc}); trying pdfplumber")

    if not text.strip():
        try:
            text = _extract_with_pdfplumber(file_path)
        except EncryptedDocumentError:
            raise
        except Exception as exc:  # noqa: BLE001
            if primary_error is not None:
                raise CorruptedDocumentError(
                    f"Could not parse PDF with either backend: {file_path} "
                    f"(PyMuPDF: {primary_error}; pdfplumber: {exc})"
                ) from exc
            raise CorruptedDocumentError(f"Could not parse PDF: {file_path} ({exc})") from exc

    if not text.strip():
        raise EmptyDocumentError(
            f"No extractable text found in {file_path}. "
            "This is likely a scanned / image-only PDF. OCR is out of scope "
            "for this pipeline — re-export the resume as a text-based PDF."
        )

    return text
