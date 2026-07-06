"""Composite candidate-vs-JD scorer."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.features import CandidateProfile
from src.ranking.gbm_model import GBMRankingModel
from src.ranking.jd_analyzer import JDRequirements, analyze_jd
from src.ranking.model import RankingModel
from src.similarity.hybrid_scorer import hybrid_similarity
from src.utils.config import get as cfg_get

DEFAULT_WEIGHTS = {
    "skills": 0.45,
    "experience": 0.30,
    "education": 0.15,
    "semantic": 0.10,
}


@dataclass
class ScoreBreakdown:
    skills: float
    experience: float
    education: float
    semantic: float
    total: float
    # Explainability — additive fields, safe for existing asdict() callers.
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    reasoning: str = ""
    model_source: str = "config-prior"
    feature_importances: dict[str, float] = field(default_factory=dict)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _skills_score(
    candidate_skills: list[str],
    required_skills: list[str],
    nice_to_have_skills: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Recall of required skills, i.e. what fraction of what the JD asks for
    does the candidate actually have — plus a small bonus for optional
    ("nice to have") skills, capped so it can't push the score past 1.0.
    Returns (score, missing_required_skills)."""
    candidate_set = set(candidate_skills)

    if not required_skills:
        # No skills could be parsed out of the JD — fall back to a mild
        # proxy so candidates aren't zeroed out through no fault of their own.
        return _clip01(len(candidate_skills) / 10), []

    required_set = set(required_skills)
    matched = candidate_set & required_set
    missing = sorted(required_set - candidate_set)
    base_score = len(matched) / len(required_set)

    bonus = 0.0
    if nice_to_have_skills:
        nice_matched = candidate_set & set(nice_to_have_skills)
        bonus = 0.1 * (len(nice_matched) / len(nice_to_have_skills))

    return _clip01(base_score + bonus), missing


def _experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        # JD doesn't specify a number — reward any experience, mildly.
        return _clip01(candidate_years / 5) if candidate_years > 0 else 0.5
    return _clip01(candidate_years / required_years)


def _education_score(candidate_level: int, required_level: int) -> float:
    if required_level <= 0:
        return 1.0 if candidate_level > 0 else 0.5
    if candidate_level >= required_level:
        return 1.0
    return candidate_level / required_level


def _build_reasoning(
    skills_score: float,
    experience_score: float,
    education_score: float,
    semantic_score: float,
    missing_skills: list[str],
    candidate_years: float,
    required_years: float,
    total: float,
) -> tuple[str, list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    if skills_score >= 0.7:
        strengths.append(f"Strong skills match ({skills_score:.0%} of required skills found)")
    elif skills_score < 0.4:
        weaknesses.append(f"Limited skills match ({skills_score:.0%} of required skills found)")

    if required_years > 0:
        if candidate_years >= required_years:
            strengths.append(
                f"Meets experience requirement ({candidate_years:.1f} yrs vs {required_years:.0f} required)"
            )
        else:
            weaknesses.append(
                f"Below required experience ({candidate_years:.1f} yrs vs {required_years:.0f} required)"
            )

    if education_score >= 1.0:
        strengths.append("Education meets or exceeds requirement")
    elif education_score < 0.6:
        weaknesses.append("Education level below requirement")

    if semantic_score >= 0.5:
        strengths.append("High overall semantic alignment with the job description")
    elif semantic_score < 0.2:
        weaknesses.append("Low overall semantic alignment with the job description")

    if missing_skills:
        weaknesses.append(f"Missing skills: {', '.join(missing_skills[:8])}")

    verdict = "a strong" if total >= 0.7 else "a moderate" if total >= 0.4 else "a weak"
    reasoning = (
        f"Overall fit score {total:.2f} — {verdict} match. "
        f"Skills {skills_score:.0%}, experience {experience_score:.0%}, "
        f"education {education_score:.0%}, semantic {semantic_score:.0%}."
    )
    return reasoning, strengths, weaknesses


def _select_model(weights: dict[str, float] | None):
    """Pick the scoring backend for this call.

    - Explicit `weights` -> always the transparent linear model (an explicit
      weights override is a request for the interpretable formula, not the
      trained ensemble).
    - `ranking.model_type: logistic` in config.yaml -> always the linear
      model, e.g. for reproducible/interpretable-only deployments that want
      to opt out of the trained ensemble entirely.
    - Otherwise (`ranking.model_type: gradient_boosting`, the default) ->
      the trained Gradient Boosting model if one has been persisted to
      `models/ranking_model.joblib` (see `src/ranking/train.py`), falling
      back to the linear config-prior if no trained artifact exists yet.
      Selection happens once per call, not once per feature — cheap either
      way since both `.load()` calls are themselves cached/lazy.
    """
    if weights is not None:
        return RankingModel(weights=weights)

    model_type = cfg_get("ranking.model_type", "gradient_boosting")
    if model_type == "gradient_boosting":
        gbm = GBMRankingModel.load()
        if gbm is not None:
            return gbm
    return RankingModel.load()


def score_candidate(
    profile: CandidateProfile,
    jd_text: str,
    jd_required_skills: list[str] | None = None,
    weights: dict[str, float] | None = None,
    jd_requirements: JDRequirements | None = None,
) -> ScoreBreakdown:
    """Compute a composite, explainable score in [0, 1].

    `jd_required_skills` / `jd_requirements` let a caller override what the
    JD analyzer would otherwise derive automatically from `jd_text` — useful
    if a caller already parsed the JD once for a whole batch and wants to
    avoid redoing it per candidate.
    """
    requirements = jd_requirements or analyze_jd(jd_text)
    required_skills = (
        jd_required_skills if jd_required_skills is not None else requirements.required_skills
    )

    skills_score, missing_skills = _skills_score(
        profile.skills, required_skills, requirements.nice_to_have_skills
    )
    experience_score = _experience_score(
        profile.total_experience_years, requirements.min_experience_years
    )

    from src.features.education_extractor import highest_degree_level

    candidate_level = highest_degree_level(profile.education)
    education_score = _education_score(candidate_level, requirements.required_degree_level)

    semantic_score = hybrid_similarity(profile.raw_text, jd_text)

    model = _select_model(weights)

    features = {
        "skills": skills_score,
        "experience": experience_score,
        "education": education_score,
        "semantic": semantic_score,
    }
    total = _clip01(model.predict(features))
    importances = model.feature_importances() if isinstance(model, GBMRankingModel) else {}

    reasoning, strengths, weaknesses = _build_reasoning(
        skills_score,
        experience_score,
        education_score,
        semantic_score,
        missing_skills,
        profile.total_experience_years,
        requirements.min_experience_years,
        total,
    )

    return ScoreBreakdown(
        skills=skills_score,
        experience=experience_score,
        education=education_score,
        semantic=semantic_score,
        total=total,
        strengths=strengths,
        weaknesses=weaknesses,
        missing_skills=missing_skills,
        reasoning=reasoning,
        model_source=model.source,
        feature_importances=importances,
    )
