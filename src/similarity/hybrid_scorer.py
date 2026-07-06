"""Combine TF-IDF (lexical) and embedding (semantic) similarity."""

from __future__ import annotations

from src.similarity.embedding_matcher import embedding_similarity
from src.similarity.tfidf_matcher import tfidf_similarity
from src.utils.config import get as cfg_get
from src.utils.exceptions import EmbeddingModelUnavailableError
from src.utils.logger import get_logger

log = get_logger(__name__)

_warned_once = False


def hybrid_similarity(
    resume_text: str,
    jd_text: str,
    tfidf_weight: float | None = None,
    embedding_weight: float | None = None,
    method: str | None = None,
) -> float:
    """Similarity between a resume and a JD, using the configured backend.

    `similarity.method` in config.yaml selects the backend:
    - `"tfidf"` — lexical only. Fast, fully offline, no ML dependency
      beyond `scikit-learn`. Misses paraphrases ("led a team" vs "managed
      engineers").
    - `"embedding"` — semantic only (`sentence-transformers`). Catches
      paraphrases TF-IDF can't, at the cost of a heavier dependency and a
      model-weights download on first use.
    - `"hybrid"` (default, recommended) — a weighted blend of both
      (`similarity.tfidf_weight` / `similarity.embedding_weight`). Lexical
      overlap and semantic similarity are complementary signals, not
      redundant ones — a resume can paraphrase every requirement (high
      embedding, low TF-IDF) or parrot the JD's exact keywords with no real
      substance behind them (high TF-IDF, misleadingly high without a
      semantic check); blending catches more of both failure modes than
      either alone. See README "Trade-offs" for the full comparison.

    Regardless of `method`, this never raises for a missing embedding
    backend — `"embedding"`/`"hybrid"` both fall back to TF-IDF-only if
    `sentence-transformers`/`torch` aren't installed or there's no network
    to fetch model weights on first use, since semantic similarity is a
    quality enhancement, not a hard requirement for the pipeline to run.
    """
    global _warned_once

    method = method or cfg_get("similarity.method", "hybrid")

    if method == "tfidf":
        return tfidf_similarity(resume_text, jd_text)

    if method == "embedding":
        try:
            return embedding_similarity(resume_text, jd_text)
        except EmbeddingModelUnavailableError as exc:
            if not _warned_once:
                log.warning(f"Embedding similarity unavailable, using TF-IDF only: {exc}")
                _warned_once = True
            return tfidf_similarity(resume_text, jd_text)

    # "hybrid" (default) and any unrecognized value fall through to the
    # weighted blend below, rather than raising on a config typo.
    tfidf_weight = (
        tfidf_weight if tfidf_weight is not None else cfg_get("similarity.tfidf_weight", 0.4)
    )
    embedding_weight = (
        embedding_weight
        if embedding_weight is not None
        else cfg_get("similarity.embedding_weight", 0.6)
    )

    lex = tfidf_similarity(resume_text, jd_text)

    try:
        sem = embedding_similarity(resume_text, jd_text)
    except EmbeddingModelUnavailableError as exc:
        if not _warned_once:
            log.warning(f"Embedding similarity unavailable, using TF-IDF only: {exc}")
            _warned_once = True
        return lex

    return tfidf_weight * lex + embedding_weight * sem
