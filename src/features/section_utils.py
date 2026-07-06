"""Detect resume section boundaries (Experience, Education, ...).

Shared by `experience_extractor` and `education_extractor` so section
detection logic lives in exactly one place instead of being duplicated
across both.
"""

from __future__ import annotations

SECTION_HEADERS: dict[str, list[str]] = {
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "career history",
        "relevant experience",
    ],
    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "educational qualifications",
        "academic details",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "key skills",
        "skills & tools",
        "skills and tools",
    ],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "certifications": [
        "certifications",
        "certificates",
        "licenses & certifications",
        "licenses and certifications",
        "certifications & courses",
    ],
    "summary": ["summary", "objective", "professional summary", "profile", "about me"],
    "publications": ["publications"],
    "awards": ["awards", "honors", "achievements", "awards & honors"],
    "languages": ["languages"],
    "references": ["references"],
}

_ALL_HEADER_PHRASES = {h for phrases in SECTION_HEADERS.values() for h in phrases}


def _is_header_line(line: str, phrases: set[str]) -> bool:
    stripped = line.strip().strip(":-").lower()
    return stripped in phrases


def extract_section(text: str, section_key: str) -> str:
    """Return the text block under `section_key`'s header, or "" if absent.

    The block runs from just after the matching header line to just before
    the next line that matches *any other* known section header (or to the
    end of the document).
    """
    if section_key not in SECTION_HEADERS:
        raise ValueError(f"Unknown section key: {section_key}")

    lines = text.splitlines()
    target_phrases = set(SECTION_HEADERS[section_key])
    other_phrases = _ALL_HEADER_PHRASES - target_phrases

    start = None
    for i, line in enumerate(lines):
        if _is_header_line(line, target_phrases):
            start = i + 1
            break
    if start is None:
        return ""

    end = len(lines)
    for j in range(start, len(lines)):
        if _is_header_line(lines[j], other_phrases):
            end = j
            break

    return "\n".join(lines[start:end]).strip()
