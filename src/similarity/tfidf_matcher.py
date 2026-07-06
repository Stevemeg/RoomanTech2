"""TF-IDF cosine similarity between a resume and a job description."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Return a cosine-similarity score in [0, 1].

    Fits a fresh TF-IDF vectorizer on just the (resume, JD) pair. This is
    cheap enough to redo per comparison and avoids the staleness/index
    problems of a corpus-wide vectorizer for what is effectively a
    one-off pairwise comparison.
    """
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()
    if not resume_text or not jd_text:
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    try:
        matrix = vectorizer.fit_transform([resume_text, jd_text])
    except ValueError:
        # Happens if, after stop-word removal, vocabulary is empty.
        return 0.0

    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(max(0.0, min(1.0, score)))
