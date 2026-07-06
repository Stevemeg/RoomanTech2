"""Normalize extracted resume text."""

import re
import unicodedata

# A standalone line that's just a page number ("3", "Page 3", "3 / 8", "- 3 -")
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\(?-?\s*\d{1,3}\s*(/\s*\d{1,3})?\)?\s*-?\s*$", re.I)

# Word broken across a line wrap by a hyphen, e.g. "develop-\nment" -> "development"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def _strip_page_numbers(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(line for line in lines if not _PAGE_NUMBER_RE.match(line))


def clean_text(text: str) -> str:
    """Strip control chars, collapse whitespace, normalize bullets/unicode.

    Order matters: unicode normalization first (so later regexes see a
    canonical form), then structural cleanup (hyphenation, page numbers),
    then whitespace collapsing last.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[•·●◦▪]", "-", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _strip_page_numbers(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
