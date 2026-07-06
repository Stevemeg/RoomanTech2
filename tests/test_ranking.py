"""Tests for the ranking pipeline: JD analysis, model, scorer, ranker."""

from src.features import CandidateProfile
from src.ranking.jd_analyzer import analyze_jd
from src.ranking.model import RankingModel
from src.ranking.ranker import rank_candidates
from src.ranking.scorer import score_candidate

SAMPLE_JD = """
Job Title: Backend Engineer
Experience: 4-7 years

Required skills:
- Python, FastAPI, PostgreSQL

Nice to have:
- Kubernetes, GraphQL

Education:
- Bachelor's degree in Computer Science
"""


def _profile(skills, years, degree="Bachelor"):
    return CandidateProfile(
        raw_text=f"Candidate with skills {' '.join(skills)} and {years} years experience.",
        skills=skills,
        education=[{"degree": degree, "institution": None, "field": None, "year": "2020"}],
        total_experience_years=years,
    )


def test_analyze_jd_splits_required_and_nice_to_have():
    req = analyze_jd(SAMPLE_JD)
    assert "python" in req.required_skills
    assert "fastapi" in req.required_skills
    assert "kubernetes" not in req.required_skills
    assert "kubernetes" in req.nice_to_have_skills
    assert req.min_experience_years == 4.0
    assert req.required_degree_level == 2


def test_ranking_model_prior_is_monotonic_with_weighted_sum():
    model = RankingModel.load()
    high = model.predict({"skills": 0.9, "experience": 0.9, "education": 0.9, "semantic": 0.9})
    low = model.predict({"skills": 0.1, "experience": 0.1, "education": 0.1, "semantic": 0.1})
    assert high > low


def test_ranking_model_fit_persists_weights(tmp_path):
    import random

    random.seed(0)
    X = [
        {
            "skills": random.random(),
            "experience": random.random(),
            "education": random.random(),
            "semantic": random.random(),
        }
        for _ in range(50)
    ]
    y = [1 if row["skills"] > 0.5 else 0 for row in X]

    save_path = tmp_path / "weights.json"
    model = RankingModel()
    model.fit(X, y, save_path=save_path)

    assert save_path.exists()
    reloaded = RankingModel.load(weights_path=save_path)
    assert reloaded.source.startswith("trained")


def test_score_candidate_flags_missing_required_skills():
    profile = _profile(["python"], years=5)
    score = score_candidate(profile, SAMPLE_JD)
    assert "fastapi" in score.missing_skills or "postgresql" in score.missing_skills
    assert score.reasoning


def test_score_candidate_full_match_scores_higher_than_partial():
    strong = _profile(["python", "fastapi", "postgresql", "kubernetes"], years=6)
    weak = _profile(["html", "css"], years=1, degree="Diploma")
    strong_score = score_candidate(strong, SAMPLE_JD)
    weak_score = score_candidate(weak, SAMPLE_JD)
    assert strong_score.total > weak_score.total


def test_rank_candidates_orders_by_total_desc_with_keys():
    strong = _profile(["python", "fastapi", "postgresql"], years=6)
    weak = _profile(["html"], years=1, degree="Diploma")
    ranked = rank_candidates([weak, strong], SAMPLE_JD, keys=["weak.txt", "strong.txt"])

    assert [key for key, _p, _s in ranked] == ["strong.txt", "weak.txt"]
    assert ranked[0][2].total >= ranked[1][2].total


def test_rank_candidates_respects_top_k():
    profiles = [_profile(["python"], years=i) for i in range(5)]
    ranked = rank_candidates(profiles, SAMPLE_JD, top_k=2)
    assert len(ranked) == 2


def test_score_candidate_uses_trained_gbm_backend_when_available():
    profile = _profile(["python", "fastapi", "postgresql"], years=6)
    score = score_candidate(profile, SAMPLE_JD)
    # The repo ships a trained GBM model (models/ranking_model.joblib) —
    # it should be preferred over the linear fallback by default.
    assert score.model_source.startswith("trained-gbm")
    assert score.feature_importances  # GBM path populates this; linear doesn't


def test_score_candidate_falls_back_to_linear_when_gbm_unavailable(monkeypatch):
    from src.ranking import scorer as scorer_module

    monkeypatch.setattr(scorer_module.GBMRankingModel, "load", staticmethod(lambda *a, **k: None))
    profile = _profile(["python", "fastapi", "postgresql"], years=6)
    score = score_candidate(profile, SAMPLE_JD)
    assert not score.model_source.startswith("trained-gbm")
    assert score.feature_importances == {}


def test_score_candidate_explicit_weights_always_use_linear_model():
    profile = _profile(["python", "fastapi", "postgresql"], years=6)
    custom_weights = {"skills": 1.0, "experience": 0.0, "education": 0.0, "semantic": 0.0}
    score = score_candidate(profile, SAMPLE_JD, weights=custom_weights)
    assert score.model_source == "config-prior"
