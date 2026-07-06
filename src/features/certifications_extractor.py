"""Extract certification entries from resume text."""

from __future__ import annotations

import re

from src.features.section_utils import extract_section

# Recognized even outside a dedicated "Certifications" section, since many
# resumes list a single certification inline under Education or Summary.
_CERT_KEYWORD_RE = re.compile(r"\b(certified|certification|certificate)\b", re.IGNORECASE)


def extract_certifications(text: str) -> list[str]:
    """Return a deduplicated list of certification lines/mentions.

    Uses the dedicated Certifications section when present (one entry per
    non-empty line); otherwise falls back to scanning the full text for
    lines mentioning "certified" / "certificate" / "certification".
    """
    section = extract_section(text, "certifications")
    if section:
        lines = [ln.strip("-• \t") for ln in section.splitlines() if ln.strip()]
    else:
        lines = [
            ln.strip("-• \t")
            for ln in text.splitlines()
            if ln.strip() and _CERT_KEYWORD_RE.search(ln)
        ]

    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            result.append(line)
    return result
