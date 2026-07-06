from pathlib import Path

from src.parser.docx_parser import parse_docx
from src.parser.pdf_parser import parse_pdf
from src.parser.text_cleaner import clean_text
from src.utils.config import get as cfg_get
from src.utils.exceptions import (
    EmptyDocumentError,
    FileTooLargeError,
    ResumeParsingError,
    ResumeRankerError,
    UnsupportedFileTypeError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

__all__ = ["parse_pdf", "parse_docx", "clean_text", "parse_resume"]


def _validate_file(file_path: str) -> str:
    """Extension + size checks, per the security requirement that we never
    attempt to parse (let alone execute) an untrusted file blindly."""
    path = Path(file_path)
    suffix = path.suffix.lower().lstrip(".")
    supported = cfg_get("parser.supported_formats", ["pdf", "docx", "txt"])
    if suffix not in supported:
        raise UnsupportedFileTypeError(
            f"Unsupported file format '.{suffix}' for {file_path}. " f"Supported: {supported}"
        )

    max_mb = cfg_get("parser.max_file_size_mb", 10)
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError as exc:
        raise ResumeParsingError(f"Cannot stat file {file_path}: {exc}") from exc
    if size_mb > max_mb:
        raise FileTooLargeError(f"{file_path} is {size_mb:.1f} MB, exceeds the {max_mb} MB limit")
    return suffix


def parse_resume(file_path: str) -> str:
    """Validate and dispatch a resume file to the right parser, then clean it.

    Raises a `ResumeRankerError` subclass (never a bare exception) so batch
    callers can catch one type and skip the file instead of crashing the run.
    """
    suffix = _validate_file(file_path)

    try:
        if suffix == "pdf":
            raw = parse_pdf(file_path)
        elif suffix == "docx":
            raw = parse_docx(file_path)
        elif suffix == "txt":
            with open(file_path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            if not raw.strip():
                raise EmptyDocumentError(f"{file_path} is empty")
        else:  # pragma: no cover - guarded by _validate_file
            raise UnsupportedFileTypeError(f"Unsupported file format: {file_path}")
    except ResumeRankerError:
        raise
    except Exception as exc:  # noqa: BLE001 - never let one bad file kill a batch
        raise ResumeParsingError(f"Unexpected error parsing {file_path}: {exc}") from exc

    return clean_text(raw)
