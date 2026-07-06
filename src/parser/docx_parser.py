"""DOCX resume parser (python-docx)."""

from __future__ import annotations

from src.utils.exceptions import CorruptedDocumentError, EmptyDocumentError
from src.utils.logger import get_logger

log = get_logger(__name__)


def parse_docx(file_path: str) -> str:
    """Extract raw text from a DOCX resume.

    Walks the document body in order so that paragraphs interleaved with
    tables (a common resume pattern: a skills table sandwiched between
    summary and experience paragraphs) come out in the right sequence,
    instead of "all paragraphs, then all tables".
    """
    try:
        import docx
        from docx.oxml.ns import qn
    except Exception as exc:  # noqa: BLE001
        raise CorruptedDocumentError(f"python-docx unavailable: {exc}") from exc

    try:
        document = docx.Document(file_path)
    except Exception as exc:  # noqa: BLE001
        raise CorruptedDocumentError(f"Could not open DOCX: {file_path} ({exc})") from exc

    chunks: list[str] = []
    body = document.element.body

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            para_text = "".join(node.text or "" for node in child.iter(qn("w:t")))
            if para_text.strip():
                chunks.append(para_text)
        elif child.tag == qn("w:tbl"):
            for row in child.iter(qn("w:tr")):
                cells = []
                for cell in row.iter(qn("w:tc")):
                    cell_text = "".join(node.text or "" for node in cell.iter(qn("w:t")))
                    if cell_text.strip():
                        cells.append(cell_text.strip())
                if cells:
                    chunks.append(" | ".join(cells))

    text = "\n".join(chunks)
    if not text.strip():
        raise EmptyDocumentError(f"No extractable text found in {file_path}.")
    return text
