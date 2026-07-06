"""Tests for the primary trained-ML (Gradient Boosting) ranking backend."""

import random

from src.ranking.gbm_model import GBMRankingModel
from src.ranking.model import FEATURE_NAMES


def _synthetic_data(n=120, seed=7):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        row = {name: rng.random() for name in FEATURE_NAMES}
        score = sum(row.values()) / len(FEATURE_NAMES) + 0.2 * row["skills"] * row["experience"]
        X.append(row)
        y.append(max(0.0, min(1.0, score)))
    return X, y


def test_load_returns_none_when_no_model_file(tmp_path):
    missing = tmp_path / "does_not_exist.joblib"
    assert GBMRankingModel.load(path=missing) is None


def test_fit_persists_and_reloads(tmp_path):
    X, y = _synthetic_data()
    save_path = tmp_path / "gbm.joblib"

    trained = GBMRankingModel.fit(X, y, save_path=save_path)
    assert save_path.exists()
    assert trained.source.startswith("trained-gbm")

    reloaded = GBMRankingModel.load(path=save_path)
    assert reloaded is not None
    assert reloaded.source.startswith("trained-gbm")


def test_predict_is_bounded_and_responsive_to_inputs(tmp_path):
    X, y = _synthetic_data()
    model = GBMRankingModel.fit(X, y, save_path=tmp_path / "gbm.joblib")

    high = model.predict({"skills": 0.9, "experience": 0.9, "education": 0.9, "semantic": 0.9})
    low = model.predict({"skills": 0.05, "experience": 0.05, "education": 0.05, "semantic": 0.05})

    assert 0.0 <= low <= high <= 1.0
    assert high > low


def test_feature_importances_are_reported_and_sum_to_about_one(tmp_path):
    X, y = _synthetic_data()
    model = GBMRankingModel.fit(X, y, save_path=tmp_path / "gbm.joblib")

    importances = model.feature_importances()
    assert set(importances.keys()) == set(FEATURE_NAMES)
    assert 0.99 <= sum(importances.values()) <= 1.01


def test_load_handles_corrupted_model_file_gracefully(tmp_path):
    bad_path = tmp_path / "corrupt.joblib"
    bad_path.write_bytes(b"not a real joblib file")
    assert GBMRankingModel.load(path=bad_path) is None
