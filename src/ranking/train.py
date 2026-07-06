"""Train (or re-train) the candidate-ranking models.

Usage:
    # Demo run — fits on a synthetic, clearly-labeled bootstrap dataset,
    # since no real historical shortlist/hire data ships with this repo.
    # Trains BOTH the primary GBM model and the linear fallback, and prints
    # a held-out benchmark comparing them.
    python -m src.ranking.train --demo

    # Real run — once you have historical labels (e.g. exported from an ATS
    # as "was this candidate shortlisted for this JD?"), point at a CSV
    # with columns: skills,experience,education,semantic,label
    python -m src.ranking.train --labels data/training/labeled_features.csv

This is the supervised half of the trade-off documented in `model.py`,
`gbm_model.py`, and the README: the pipeline runs perfectly well with zero
labeled data (falling back to the config-driven linear prior), and this
script is the on-ramp to genuinely learned models — both of them — the
moment labels exist. It never runs implicitly during scoring; scoring only
ever *loads* what this script persisted.
"""

from __future__ import annotations

import argparse
import csv
import random

from src.ranking.gbm_model import DEFAULT_GBM_MODEL_PATH, FEATURE_NAMES, GBMRankingModel
from src.ranking.model import DEFAULT_WEIGHTS_PATH, RankingModel
from src.utils.logger import get_logger

log = get_logger(__name__)

_PRIOR = {"skills": 0.45, "experience": 0.30, "education": 0.15, "semantic": 0.10}
# How much extra credit a candidate who is strong on BOTH skills and
# experience gets, on top of the plain weighted sum — this non-linear
# interaction is exactly what the linear model structurally cannot learn,
# and what motivates choosing Gradient Boosting as the primary backend
# (see README "Model Architecture").
_INTERACTION_WEIGHT = 0.35


def _synthetic_true_score(row: dict[str, float]) -> float:
    linear = sum(_PRIOR[name] * row[name] for name in FEATURE_NAMES)
    interaction = _INTERACTION_WEIGHT * row["skills"] * row["experience"]
    return linear + interaction


def _make_synthetic_dataset(n: int, seed: int) -> tuple[list[dict], list[float], list[int]]:
    """Generate a plausible bootstrap dataset for demo purposes only.

    Returns (features, continuous_target, binary_label). The continuous
    target feeds the GBM regressor; the binary label feeds the linear
    model's logistic fit — both derived from the same underlying synthetic
    "true" score plus independent noise per model, so neither model sees
    the other's exact target.

    This demonstrates the training *mechanism* end-to-end and is explicitly
    NOT real hiring data — replace with `--labels real_data.csv` as soon as
    real labels exist.
    """
    rng = random.Random(seed)
    X, y_continuous, y_binary = [], [], []
    for _ in range(n):
        row = {name: rng.random() for name in FEATURE_NAMES}
        true_score = _synthetic_true_score(row)
        X.append(row)
        y_continuous.append(max(0.0, min(1.0, true_score + rng.gauss(0, 0.05))))
        y_binary.append(1 if true_score + rng.gauss(0, 0.07) >= 0.5 else 0)
    return X, y_continuous, y_binary


def _load_csv_dataset(path: str) -> tuple[list[dict], list[float], list[int]]:
    X, y_continuous, y_binary = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append({name: float(row[name]) for name in FEATURE_NAMES})
            label = float(row["label"])
            y_continuous.append(label)
            y_binary.append(1 if label >= 0.5 else 0)
    return X, y_continuous, y_binary


def _train_test_split(
    X: list[dict], y_continuous: list[float], y_binary: list[int], test_frac: float, seed: int
):
    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * (1 - test_frac))
    train_idx, test_idx = indices[:split], indices[split:]

    def _take(idx):
        return [X[i] for i in idx], [y_continuous[i] for i in idx], [y_binary[i] for i in idx]

    return _take(train_idx), _take(test_idx)


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation, computed without adding a scipy
    dependency for one metric — ranks two equal-length lists and returns
    Pearson correlation of the ranks."""
    n = len(a)

    def _ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank, i in enumerate(order):
            ranks[i] = rank
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a**0.5 * var_b**0.5)


def _benchmark(
    test_X: list[dict], test_true: list[float], gbm: GBMRankingModel, linear: RankingModel
) -> None:
    """Rank-correlate each model's predictions against the synthetic
    "true" score on held-out data. This is an honest number computed from
    the synthetic bootstrap, presented as exactly that — not a claim about
    real-world hiring accuracy, which requires real labels to measure."""
    gbm_preds = [gbm.predict(row) for row in test_X]
    linear_preds = [linear.predict(row) for row in test_X]

    gbm_corr = _spearman(gbm_preds, test_true)
    linear_corr = _spearman(linear_preds, test_true)

    log.info(
        f"Held-out rank correlation vs synthetic true score — "
        f"GBM: {gbm_corr:.3f}, Linear: {linear_corr:.3f} "
        f"(n={len(test_X)}; synthetic data, not a real-world accuracy claim)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the candidate-ranking models")
    parser.add_argument("--labels", help="CSV with skills,experience,education,semantic,label")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic demo dataset")
    parser.add_argument("--n", type=int, default=400, help="Synthetic dataset size (--demo only)")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gbm-output", default=str(DEFAULT_GBM_MODEL_PATH))
    parser.add_argument("--linear-output", default=str(DEFAULT_WEIGHTS_PATH))
    args = parser.parse_args()

    if args.labels:
        X, y_continuous, y_binary = _load_csv_dataset(args.labels)
        log.info(f"Loaded {len(X)} labeled examples from {args.labels}")
    elif args.demo:
        X, y_continuous, y_binary = _make_synthetic_dataset(n=args.n, seed=args.seed)
        log.info(f"Generated {len(X)} synthetic demo examples (not real hiring data)")
    else:
        parser.error("Provide --labels <csv> for real data, or --demo for a synthetic run")
        return

    (train_X, train_y_cont, train_y_bin), (test_X, test_y_cont, _test_y_bin) = _train_test_split(
        X, y_continuous, y_binary, test_frac=args.test_frac, seed=args.seed
    )

    gbm = GBMRankingModel.fit(train_X, train_y_cont, save_path=args.gbm_output)
    log.info(f"GBM feature importances: {gbm.feature_importances()}")

    linear = RankingModel()
    linear.fit(train_X, train_y_bin, save_path=args.linear_output)
    log.info(f"Linear weights: {linear.weights}")

    if test_X:
        _benchmark(test_X, test_y_cont, gbm, linear)

    log.info(f"Done. GBM -> {args.gbm_output} | Linear -> {args.linear_output}")


if __name__ == "__main__":
    main()
