"""Sentence-transformer based semantic similarity."""

from __future__ import annotations

from src.utils.config import get as cfg_get
from src.utils.exceptions import EmbeddingModelUnavailableError
from src.utils.logger import get_logger

log = get_logger(__name__)

# Cache loaded models by name at module level so repeated calls (e.g. one
# resume batch scored against the same JD) don't reload the model each time.
_MODEL_CACHE: dict[str, object] = {}

# Per-model, per-text embedding cache: encoding the JD once per batch (not
# once per candidate) is the main win, since the same JD text is passed in
# for every candidate in a ranking run.
_EMBEDDING_CACHE: dict[str, dict[tuple[str, str], object]] = {}

# Long resumes get truncated to roughly this many characters before encoding
# — most transformer encoders have a token limit (e.g. 256 tokens for
# all-MiniLM-L6-v2) and silently truncate anyway; doing it explicitly keeps
# behaviour predictable and keeps us from cutting off the JD instead of the
# (usually longer) resume.
_MAX_CHARS = 4000


def _load_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingModelUnavailableError("sentence-transformers is not installed") from exc

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - e.g. no internet to fetch weights
        raise EmbeddingModelUnavailableError(
            f"Could not load embedding model '{model_name}': {exc}"
        ) from exc

    _MODEL_CACHE[model_name] = model
    return model


def _encode_cached(model, model_name: str, text: str):
    """Cache embeddings per (model_name, text) so scoring many candidates
    against the same JD only encodes that JD once."""
    cache_key = (model_name, text)
    cache = _EMBEDDING_CACHE.setdefault(model_name, {})
    if cache_key not in cache:
        cache[cache_key] = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        # Simple bound so a very large batch doesn't grow this unbounded.
        if len(cache) > 512:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


def embedding_similarity(resume_text: str, jd_text: str, model_name: str | None = None) -> float:
    """Return semantic similarity in [0, 1] using sentence embeddings.

    Raises `EmbeddingModelUnavailableError` (never a bare crash) if the
    model can't be loaded — e.g. `sentence-transformers`/`torch` aren't
    installed, or there's no network access to fetch model weights on
    first use. `hybrid_scorer` is the intended caller and knows how to fall
    back to TF-IDF-only similarity when this happens.
    """
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()
    if not resume_text or not jd_text:
        return 0.0

    name = model_name or cfg_get(
        "similarity.embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = _load_model(name)

    import numpy as np

    resume_vec = _encode_cached(model, name, resume_text[:_MAX_CHARS])
    jd_vec = _encode_cached(model, name, jd_text[:_MAX_CHARS])
    score = float(np.dot(resume_vec, jd_vec))
    return max(0.0, min(1.0, score))
