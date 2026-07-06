"""Centralized, cached access to config/config.yaml.

Every module that needs a tunable value (thresholds, weights, model names,
date formats, ...) should read it from here instead of hardcoding it, so the
whole pipeline stays configurable from a single YAML file.

`.env` (see `.env.example`) can override a small, documented set of values
without editing config.yaml — handy for per-environment overrides (e.g. a
different embedding model in CI vs. locally) without touching version-controlled
config.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

load_dotenv()  # no-op if .env doesn't exist — safe to call unconditionally

# dotted config key -> environment variable name, mirroring .env.example.
_ENV_OVERRIDES: dict[str, str] = {
    "similarity.embedding_model": "EMBEDDING_MODEL",
    "nlp.spacy_model": "SPACY_MODEL",
    "ranking.weights.skills": "WEIGHT_SKILLS",
    "ranking.weights.experience": "WEIGHT_EXPERIENCE",
    "ranking.weights.education": "WEIGHT_EDUCATION",
    "ranking.weights.semantic": "WEIGHT_SEMANTIC",
}


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache config.yaml. Returns {} if the file is missing.

    Cached with lru_cache so the file is parsed once per process. Tests that
    need a different config can call `load_config.cache_clear()` first.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested config value using dot notation.

    Example: get("similarity.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    Checks `_ENV_OVERRIDES` first so a value from `.env` wins over
    config.yaml, then falls back to config.yaml, then to `default`.
    """
    env_var = _ENV_OVERRIDES.get(dotted_key)
    if env_var and (env_value := os.getenv(env_var)) is not None:
        return _coerce_like(env_value, default)

    cfg = load_config()
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def _coerce_like(value: str, default: Any) -> Any:
    """Cast an env var string to match the type of `default` (float weights
    in particular need to come back as floats, not strings)."""
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    return value
