"""Extract work-experience entries from resume text."""

from __future__ import annotations

import re
from datetime import datetime

from src.features.section_utils import extract_section
from src.utils.logger import get_logger

log = get_logger(__name__)

_ONGOING_WORDS = r"present|current|currently|till\s*date|to\s*date|now|ongoing"

_DATE_TOKEN = r"(?:[A-Za-z]{3,9}\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4})"

_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_TOKEN})\s*(?:-|–|—|to|until)\s*"
    rf"(?P<end>{_DATE_TOKEN}|{_ONGOING_WORDS})",
    re.IGNORECASE,
)

# "3+ years", "5 years of experience", "over 4 years" — used as a fallback
# total when no dated entries can be parsed out of an Experience section.
_YEARS_SUMMARY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*(?:experience|exp\.?)?", re.IGNORECASE
)

# Canonical employment types, checked in this order so "Contract-to-hire"
# style text doesn't get miscategorized by a shorter substring match.
_EMPLOYMENT_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Internship", re.compile(r"\bintern(?:ship)?\b", re.IGNORECASE)),
    ("Freelance", re.compile(r"\bfreelance(?:r)?\b", re.IGNORECASE)),
    ("Contract", re.compile(r"\bcontract(?:or)?\b", re.IGNORECASE)),
    ("Part-time", re.compile(r"\bpart[- ]time\b", re.IGNORECASE)),
    ("Full-time", re.compile(r"\bfull[- ]time\b", re.IGNORECASE)),
]


def _detect_employment_type(context: str) -> str | None:
    """Best-effort employment type from the entry's title line + a couple
    of neighbouring lines. Returns `None` (not "Full-time") when unstated —
    guessing a default here would misrepresent resumes that simply didn't
    say, rather than genuinely being full-time roles."""
    for label, pattern in _EMPLOYMENT_TYPE_PATTERNS:
        if pattern.search(context):
            return label
    return None


def _parse_date_token(token: str) -> datetime | None:
    token = token.strip()
    if re.match(_ONGOING_WORDS, token, re.IGNORECASE):
        return datetime.now()
    try:
        from dateutil import parser as dateutil_parser

        return dateutil_parser.parse(token, default=datetime(1900, 1, 1))
    except Exception:  # noqa: BLE001
        log.debug(f"Could not parse date token: {token!r}")
        return None


def _guess_title_line(block_lines: list[str], date_line_idx: int) -> str:
    """The role/company usually sits on the same line as the dates, or on
    the line immediately above it."""
    same_line = block_lines[date_line_idx]
    cleaned = _DATE_RANGE_RE.sub("", same_line).strip(" -|,\t")
    if cleaned:
        return cleaned
    if date_line_idx > 0:
        return block_lines[date_line_idx - 1].strip(" -|,\t")
    return ""


def _split_role_company(title_line: str) -> tuple[str | None, str | None]:
    """Best-effort split of a "Role at Company" / "Role, Company" /
    "Role | Company" line into (role, company)."""
    if not title_line:
        return None, None
    for sep in (" at ", " @ ", " - ", " – ", " | ", ", "):
        if sep in title_line:
            left, right = title_line.split(sep, 1)
            return left.strip() or None, right.strip() or None
    return title_line.strip() or None, None


def _years_between(start: datetime, end: datetime) -> float:
    days = max((end - start).days, 0)
    return round(days / 365.25, 2)


def extract_experience(text: str) -> list[dict]:
    """Return a list of {company, role, start, end, duration_years, description}.

    Strategy: locate the Experience section (falls back to the whole
    document if no explicit header exists — some resumes bury a single job
    under a generic "Summary"), then scan line-by-line for date ranges.
    Each match's own line (or the line above it) is treated as the
    "Role at Company" title for that entry.
    """
    section = extract_section(text, "experience") or text
    lines = section.splitlines()

    entries: list[dict] = []
    for i, line in enumerate(lines):
        match = _DATE_RANGE_RE.search(line)
        if not match:
            continue

        start_dt = _parse_date_token(match.group("start"))
        end_dt = _parse_date_token(match.group("end"))
        if start_dt is None:
            continue
        end_dt = end_dt or start_dt

        title_line = _guess_title_line(lines, i)
        role, company = _split_role_company(title_line)

        # A couple of following lines, often bullet points, as description.
        description_lines = [ln.strip("- ").strip() for ln in lines[i + 1 : i + 4] if ln.strip()]

        employment_context = " ".join(
            [title_line, *lines[max(0, i - 1) : i], *lines[i + 1 : i + 3]]
        )
        employment_type = _detect_employment_type(employment_context)

        entries.append(
            {
                "role": role,
                "company": company,
                "start": match.group("start").strip(),
                "end": match.group("end").strip(),
                "duration_years": _years_between(start_dt, end_dt),
                "description": " ".join(description_lines) or None,
                "employment_type": employment_type,
            }
        )

    return entries


def compute_total_experience_years(entries: list[dict], full_text: str) -> float:
    """Sum entry durations; if no entries were parsed, fall back to a
    "X years of experience" style summary line anywhere in the resume.

    Note: overlapping roles (e.g. concurrent freelance + full-time) are
    summed rather than merged — a documented limitation, see README.
    """
    if entries:
        return round(sum(e["duration_years"] for e in entries), 2)

    match = _YEARS_SUMMARY_RE.search(full_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0
