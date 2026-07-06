"""The primary trained-ML inference backend for candidate scoring.

**Why Gradient Boosting, and why it's layered on top of `model.py`'s linear
model rather than replacing it** — see the README "Model Architecture"
section for the full write-up. Short version: `sklearn.ensemble.
GradientBoostingRegressor` captures non-linear interactions between the four
engineered features (e.g. a candidate who is strong on skills AND experience
should score disproportionately higher than the sum of two "good" scores
would suggest) that a linear/logistic formula structurally cannot, while
staying inside `scikit-learn` — already a project dependency — instead of
pulling in XGBoost/LightGBM's larger, compiled-binary footprint for a
four-feature problem that doesn't need it.

**This module never trains during inference.** `GBMRankingModel.load()`
deserializes a `.joblib` file written ahead of time by `src/ranking/train.py`
and does nothing else; `predict()` only calls `estimator.predict()`. If no
trained model file is present (e.g. a fresh clone before anyone has run
`train.py`, or `joblib`/this file went missing), `load()` returns `None` and
`scorer.py` falls back to the transparent linear prior in `model.py` — the
pipeline never breaks for lack of a trained artifact, it just gets less
sophisticated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ranking.model import FEATURE_NAMES
from src.utils.logger import get_logger

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GBM_MODEL_PATH = _REPO_ROOT / "models" / "ranking_model.joblib"


class GBMRankingModel:
    """Thin, explicit wrapper around a fitted `GradientBoostingRegressor`.

    Deliberately not a dataclass like `RankingModel` — the thing being
    wrapped is an opaque fitted sklearn estimator, not a small set of
    human-editable numbers, so a plain class with an explicit `estimator`
    attribute is clearer than trying to make it look config-like.
    """

    def __init__(self, estimator: Any, source: str) -> None:
        self.estimator = estimator
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> GBMRankingModel | None:
        """Deserialize a previously-trained model. Returns `None` (not an
        exception) if unavailable, so callers can fall back cleanly —
        an untrained pipeline is a normal, expected state, not an error."""
        model_path = Path(path) if path else DEFAULT_GBM_MODEL_PATH
        if not model_path.exists():
            return None
        try:
            import joblib

            estimator = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Could not load trained GBM model from {model_path} ({exc})")
            return None
        return cls(estimator=estimator, source=f"trained-gbm:{model_path.name}")

    def predict(self, features: dict[str, float]) -> float:
        """Score a single candidate's feature vector -> [0, 1]."""
        row = [[features.get(name, 0.0) for name in FEATURE_NAMES]]
        score = float(self.estimator.predict(row)[0])
        return max(0.0, min(1.0, score))

    def feature_importances(self) -> dict[str, float]:
        """Expose `feature_importances_` for explainability — *what the
        model learned matters more*, not just the four numbers, unlike the
        linear model where the weights are directly the importances."""
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return {}
        return dict(zip(FEATURE_NAMES, (float(v) for v in importances)))

    @classmethod
    def fit(
        cls,
        X: list[dict[str, float]],
        y: list[float],
        save_path: str | Path | None = None,
        **gbm_kwargs: Any,
    ) -> GBMRankingModel:
        """Fit a `GradientBoostingRegressor` on labeled (feature_vector,
        outcome) pairs and persist it. `y` can be binary (shortlisted=1/0)
        or a continuous target (e.g. an interview-panel score normalized to
        [0, 1]) — regression handles both, and keeps the model's output a
        smooth score rather than a hard classification.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        params: dict[str, Any] = {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "random_state": 42,
        }
        params.update(gbm_kwargs)

        feature_matrix = [[row.get(name, 0.0) for name in FEATURE_NAMES] for row in X]
        estimator = GradientBoostingRegressor(**params)
        estimator.fit(feature_matrix, y)

        path = Path(save_path) if save_path else DEFAULT_GBM_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(estimator, path)
        log.info(f"Trained GradientBoostingRegressor saved to {path}")

        return cls(estimator=estimator, source=f"trained-gbm:{path.name}")
