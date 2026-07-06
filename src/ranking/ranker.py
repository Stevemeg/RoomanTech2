"""Rank a batch of candidates against a single job description."""

from __future__ import annotations

from typing import Any, overload

from src.features import CandidateProfile
from src.ranking.jd_analyzer import analyze_jd
from src.ranking.scorer import ScoreBreakdown, score_candidate

# Sort key priority when total scores tie: skills first (the highest-weighted
# component by default), then semantic, then experience, then education —
# avoids arbitrary/order-dependent tie resolution.
_TIE_BREAK_ORDER = ("skills", "semantic", "experience", "education")


def _sort_key(score: ScoreBreakdown) -> tuple:
    return (score.total,) + tuple(getattr(score, name) for name in _TIE_BREAK_ORDER)


@overload
def rank_candidates(
    profiles: list[CandidateProfile],
    jd_text: str,
    jd_required_skills: list[str] | None = None,
    top_k: int | None = None,
    keys: None = None,
) -> list[tuple[CandidateProfile, ScoreBreakdown]]: ...


@overload
def rank_candidates(
    profiles: list[CandidateProfile],
    jd_text: str,
    jd_required_skills: list[str] | None = None,
    top_k: int | None = None,
    *,
    keys: list[Any],
) -> list[tuple[Any, CandidateProfile, ScoreBreakdown]]: ...


def rank_candidates(
    profiles: list[CandidateProfile],
    jd_text: str,
    jd_required_skills: list[str] | None = None,
    top_k: int | None = None,
    keys: list[Any] | None = None,
):
    """Score every candidate and return them sorted by total score (desc),
    with meaningful (not arbitrary) tie-breaking.

    The JD is analyzed exactly once for the whole batch (not once per
    candidate) — required skills / experience / education only depend on
    the JD, not the candidate, so recomputing them per candidate would be
    pure waste for a batch of any size.

    If `keys` is given (e.g. the source file path for each profile), returns
    `(key, profile, score)` triples instead of `(profile, score)` pairs, so
    callers can recover which candidate came from which file without
    relying on fragile positional/equality lookups.
    """
    requirements = analyze_jd(jd_text)

    scored = [
        score_candidate(p, jd_text, jd_required_skills, jd_requirements=requirements)
        for p in profiles
    ]

    if keys is not None:
        if len(keys) != len(profiles):
            raise ValueError("keys and profiles must be the same length")
        combined = list(zip(keys, profiles, scored))
        combined.sort(key=lambda row: _sort_key(row[2]), reverse=True)
        return combined[:top_k] if top_k else combined

    pairs = list(zip(profiles, scored))
    pairs.sort(key=lambda row: _sort_key(row[1]), reverse=True)
    return pairs[:top_k] if top_k else pairs
