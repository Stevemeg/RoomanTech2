"""The linear scoring backend — a transparent, config-driven fallback.

**Relationship to `gbm_model.py`:** this module is *not* the primary
scoring model. `src/ranking/gbm_model.py`'s `GBMRankingModel` (gradient
boosting) is the primary, trained-ML inference path; this module is the
cold-start-safe fallback used when no trained GBM artifact exists yet, and
the transparent/interpretable option when a caller explicitly passes
`weights=` to `score_candidate()`. See `gbm_model.py`'s module docstring
and the README's "Model Architecture" section for the full comparison.

`total = sigmoid(scale * (w·x - midpoint))`, with `w` defaulting to
`ranking.weights` in config.yaml. Because `sigmoid` is monotonic, this
produces the *exact same ranking order* as the plain weighted sum `w · x`
— so this backend's default behaviour is identical to a transparent
weighted-sum formula until real training data (`fit()`) lets the
coefficients diverge from the config prior.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import get as cfg_get
from src.utils.logger import get_logger

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "models" / "ranking_weights.json"

FEATURE_NAMES = ("skills", "experience", "education", "semantic")


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


@dataclass
class RankingModel:
    """A logistic-regression-style scorer: total = sigmoid(scale * (w·x - midpoint))."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "skills": 0.45,
            "experience": 0.30,
            "education": 0.15,
            "semantic": 0.10,
        }
    )
    scale: float = 8.0
    midpoint: float = 0.5
    source: str = "config-prior"

    @classmethod
    def load(cls, weights_path: str | Path | None = None) -> RankingModel:
        """Load trained weights from disk if they exist, else fall back to
        the config-driven prior (so the pipeline always works out of the box,
        with or without a trained model file present)."""
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
                return cls(
                    weights=payload["weights"],
                    scale=payload.get("scale", 8.0),
                    midpoint=payload.get("midpoint", 0.5),
                    source=f"trained:{path.name}",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    f"Could not load trained weights from {path} ({exc}); using config prior"
                )

        weights = cfg_get(
            "ranking.weights",
            {"skills": 0.45, "experience": 0.30, "education": 0.15, "semantic": 0.10},
        )
        weights = dict(weights)
        for name in FEATURE_NAMES:
            weights[name] = cfg_get(f"ranking.weights.{name}", weights.get(name, 0.0))
        return cls(weights=weights, source="config-prior")

    def predict(self, features: dict[str, float]) -> float:
        """Score a single candidate's feature vector -> [0, 1]."""
        linear = sum(
            self.weights.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_NAMES
        )
        return _sigmoid(self.scale * (linear - self.midpoint))

    def fit(
        self,
        X: list[dict[str, float]],
        y: list[int],
        save_path: str | Path | None = None,
    ) -> RankingModel:
        """Fit a real `sklearn.linear_model.LogisticRegression` on labeled
        (feature_vector, shortlisted/hired) examples and persist the
        resulting coefficients as this model's new weights.

        The learned coefficients are rescaled to sum to 1 (like the config
        prior) purely so they stay directly comparable/interpretable
        alongside `ranking.weights` in config.yaml — the ranking *order*
        LogisticRegression itself produces is unaffected by this rescaling.
        """
        from sklearn.linear_model import LogisticRegression

        feature_matrix = [[row.get(name, 0.0) for name in FEATURE_NAMES] for row in X]
        clf = LogisticRegression()
        clf.fit(feature_matrix, y)

        raw_coefs = clf.coef_[0]
        total = float(sum(abs(c) for c in raw_coefs)) or 1.0
        learned_weights = {
            name: round(float(coef) / total, 6) for name, coef in zip(FEATURE_NAMES, raw_coefs)
        }

        self.weights = learned_weights
        self.source = "trained:logistic_regression"

        path = Path(save_path) if save_path else DEFAULT_WEIGHTS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"weights": self.weights, "scale": self.scale, "midpoint": self.midpoint},
                f,
                indent=2,
            )
        log.info(f"Trained weights saved to {path}: {self.weights}")
        return self
