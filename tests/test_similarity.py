"""Tests for similarity scorers."""

import pytest

from src.similarity import tfidf_similarity
from src.utils.exceptions import EmbeddingModelUnavailableError


def test_tfidf_identical_text_is_one():
    text = "python developer with django experience"
    assert tfidf_similarity(text, text) == pytest.approx(1.0)


def test_tfidf_unrelated_text_is_low():
    score = tfidf_similarity(
        "python backend engineer with django and postgresql",
        "looking for a watercolor painting instructor",
    )
    assert score < 0.2


def test_tfidf_empty_input_is_zero():
    assert tfidf_similarity("", "something") == 0.0
    assert tfidf_similarity("something", "") == 0.0


def test_hybrid_similarity_falls_back_to_tfidf_when_embeddings_unavailable(monkeypatch):
    from src.similarity import hybrid_scorer

    def raise_unavailable(*args, **kwargs):
        raise EmbeddingModelUnavailableError("no torch in this environment")

    monkeypatch.setattr(hybrid_scorer, "embedding_similarity", raise_unavailable)
    text = "python developer with django experience"
    # With embeddings unavailable, hybrid should equal plain TF-IDF (weight 1.0 on lexical).
    assert hybrid_scorer.hybrid_similarity(text, text) == pytest.approx(
        tfidf_similarity(text, text)
    )


def test_similarity_method_tfidf_never_touches_embeddings(monkeypatch):
    from src.similarity import hybrid_scorer

    def boom(*args, **kwargs):
        raise AssertionError("embedding_similarity should not be called when method='tfidf'")

    monkeypatch.setattr(hybrid_scorer, "embedding_similarity", boom)
    text = "python developer with django experience"
    assert hybrid_scorer.hybrid_similarity(text, text, method="tfidf") == pytest.approx(1.0)


def test_similarity_method_embedding_falls_back_to_tfidf_when_unavailable(monkeypatch):
    from src.similarity import hybrid_scorer

    def raise_unavailable(*args, **kwargs):
        raise EmbeddingModelUnavailableError("no torch in this environment")

    monkeypatch.setattr(hybrid_scorer, "embedding_similarity", raise_unavailable)
    text = "python developer with django experience"
    assert hybrid_scorer.hybrid_similarity(text, text, method="embedding") == pytest.approx(1.0)


def test_similarity_method_is_config_driven(monkeypatch):
    from src.similarity import hybrid_scorer

    def boom(*args, **kwargs):
        raise AssertionError("embedding_similarity should not be called")

    monkeypatch.setattr(hybrid_scorer, "embedding_similarity", boom)
    monkeypatch.setattr(hybrid_scorer, "cfg_get", lambda key, default=None: "tfidf")
    text = "python developer with django experience"
    # No explicit method= passed — should read "tfidf" from (mocked) config.
    assert hybrid_scorer.hybrid_similarity(text, text) == pytest.approx(1.0)
