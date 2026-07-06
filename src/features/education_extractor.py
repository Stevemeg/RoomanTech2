"""Extract education entries from resume text."""

from __future__ import annotations

import re

from src.features.section_utils import extract_section
from src.utils.config import get as cfg_get

_INSTITUTION_HINT_RE = re.compile(
    r"([A-Z][A-Za-z.&'\- ]*(?:University|Institute|College|School)[A-Za-z.&'\- ]*)"
)
_FIELD_AFTER_IN_RE = re.compile(r"\bin\s+([A-Za-z][A-Za-z&/ ]{2,60})", re.IGNORECASE)


def _degree_pattern() -> re.Pattern:
    keywords = cfg_get(
        "features.education.degree_keywords",
        ["bachelor", "master", "phd", "doctorate", "mba", "btech", "mtech"],
    )
    # Also cover common punctuated variants (B.Tech, M.Tech, B.E., M.E., B.Sc, M.Sc)
    extra = [
        "b\\.?tech",
        "m\\.?tech",
        "b\\.?e\\.?",
        "m\\.?e\\.?",
        "b\\.?sc",
        "m\\.?sc",
        "ph\\.?d\\.?",
    ]
    pattern = "|".join(re.escape(k) for k in keywords) + "|" + "|".join(extra)
    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


def _extract_year(line: str) -> str | None:
    # Prefer the *last* 4-digit year in the window (graduation year in a
    # "2018 - 2022" range, rather than the enrollment year).
    years = re.findall(r"\b(?:19|20)\d{2}\b", line)
    return years[-1] if years else None


def extract_education(text: str) -> list[dict]:
    """Return a list of {degree, institution, field, year} dicts.

    Scans the Education section (or the whole document if no header is
    present) line by line; any line containing a recognized degree keyword
    becomes one entry, with institution/field/year filled in on a
    best-effort basis from that line and its immediate neighbours.
    """
    section = extract_section(text, "education") or text
    degree_re = _degree_pattern()

    entries: list[dict] = []
    lines = [ln for ln in section.splitlines() if ln.strip()]

    for i, line in enumerate(lines):
        degree_match = degree_re.search(line)
        if not degree_match:
            continue

        prev_line = lines[i - 1] if i > 0 else ""

        institution_match = _INSTITUTION_HINT_RE.search(line) or _INSTITUTION_HINT_RE.search(
            prev_line
        )
        field_match = _FIELD_AFTER_IN_RE.search(line)
        year = _extract_year(line) or _extract_year(prev_line)

        entries.append(
            {
                "degree": degree_match.group(0).strip(),
                "institution": institution_match.group(0).strip() if institution_match else None,
                "field": field_match.group(1).strip() if field_match else None,
                "year": year,
            }
        )

    return entries


def highest_degree_level(entries: list[dict]) -> int:
    """Map the candidate's education entries to the highest rung on the
    configured degree ladder (0 if no recognizable degree was found)."""
    levels = cfg_get("features.education.degree_levels", {})
    best = 0
    for entry in entries:
        degree = (entry.get("degree") or "").lower().replace(".", "")
        for key, level in levels.items():
            if key.replace(".", "") == degree and level > best:
                best = level
    return best
