# Shipped model artifacts

Two files are committed here on purpose, both produced by
`python -m src.ranking.train --demo`:

- **`ranking_model.joblib`** — a fitted `sklearn.ensemble.
  GradientBoostingRegressor`. This is the **primary** scoring backend;
  `src/ranking/scorer.py` loads it at runtime and never trains during
  inference.
- **`ranking_weights.json`** — a fitted logistic-regression weight vector.
  This is the **fallback** backend, used automatically if the GBM file is
  ever missing/unreadable, and always used when a caller passes an explicit
  `weights=` override (asking for the transparent linear formula on
  purpose).

## ⚠️ These were trained on synthetic data, not real hiring outcomes

No labeled dataset of real shortlist/hire decisions ships with (or exists
for) this assignment — that's a cold-start problem inherent to any
brand-new pipeline, not something to paper over. `src/ranking/train.py
--demo` generates a synthetic bootstrap dataset from the same domain-expert
prior weights `config.yaml` already documents (`ranking.weights`), plus a
deliberate skills×experience interaction term and Gaussian noise, and fits
both models against it. The held-out benchmark that training run prints
(rank correlation of each model's predictions against the synthetic "true"
score) is an honest number **about the synthetic data**, not a claim about
real-world ranking accuracy — see the README's "Trade-offs" section.

## Retraining on real data

The moment real labels exist (e.g. exported from an ATS as "was this
candidate shortlisted for this JD?"), point `train.py` at them instead:

```bash
python -m src.ranking.train --labels data/training/labeled_features.csv
```

CSV columns: `skills,experience,education,semantic,label` (the same four
engineered features `src/ranking/scorer.py` already computes, plus a
0/1 or continuous outcome). This overwrites both files above with weights
learned from real signal — no code changes needed anywhere else in the
pipeline.
