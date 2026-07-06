"""Parse a job description into structured requirements.

Reuses `extract_skills` (same taxonomy the resumes are matched against, so
skill names line up on both sides) rather than inventing a second, parallel
keyword-extraction path for JDs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.features.education_extractor import highest_degree_level
from src.features.skills_extractor import extract_skills

# "4-7 years", "4+ years", "at least 3 years", "minimum of 5 years"
_YEARS_RANGE_RE = re.compile(r"(\d+)\s*(?:-|to)\s*(\d+)\s*\+?\s*years?", re.IGNORECASE)
_YEARS_PLUS_RE = re.compile(r"(\d+)\s*\+\s*years?", re.IGNORECASE)
_YEARS_MIN_RE = re.compile(r"(?:minimum|at least|min\.?)\s*(?:of\s*)?(\d+)\s*years?", re.IGNORECASE)
_YEARS_PLAIN_RE = re.compile(r"(\d+)\s*years?", re.IGNORECASE)


_NICE_TO_HAVE_HEADER_RE = re.compile(
    r"(?im)^\s*(nice[- ]to[- ]have|preferred(?: skills| qualifications)?|good[- ]to[- ]have|bonus)s?\s*:?\s*$"
)


@dataclass
class JDRequirements:
    required_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    required_degree_level: int = 0  # 0 = not specified / any


def _extract_min_years(jd_text: str) -> float:
    match = _YEARS_RANGE_RE.search(jd_text)
    if match:
        return float(match.group(1))
    match = _YEARS_PLUS_RE.search(jd_text)
    if match:
        return float(match.group(1))
    match = _YEARS_MIN_RE.search(jd_text)
    if match:
        return float(match.group(1))
    match = _YEARS_PLAIN_RE.search(jd_text)
    if match:
        return float(match.group(1))
    return 0.0


def analyze_jd(jd_text: str) -> JDRequirements:
    """Extract required skills / minimum experience / degree level from a JD.

    Skills mentioned under a "Nice to have" / "Preferred" header are split
    out from the hard requirements, so a candidate missing a bonus skill
    isn't penalized the same as missing a genuinely required one.
    """
    header_match = _NICE_TO_HAVE_HEADER_RE.search(jd_text)
    if header_match:
        required_block = jd_text[: header_match.start()]
        nice_block = jd_text[header_match.end() :]
    else:
        required_block, nice_block = jd_text, ""

    required_skills = extract_skills(required_block)
    nice_to_have_skills = [s for s in extract_skills(nice_block) if s not in required_skills]

    min_years = _extract_min_years(jd_text)

    # Reuse the education extractor's degree-keyword matching against the
    # JD's own text (a JD lists a required degree the same way a resume
    # lists an earned one, e.g. "Bachelor's degree in Computer Science").
    from src.features.education_extractor import extract_education

    degree_entries = extract_education(jd_text)
    degree_level = highest_degree_level(degree_entries)

    return JDRequirements(
        required_skills=required_skills,
        nice_to_have_skills=nice_to_have_skills,
        min_experience_years=min_years,
        required_degree_level=degree_level,
    )
