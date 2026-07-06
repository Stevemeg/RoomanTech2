"""Extract skills from resume text using a curated taxonomy + fuzzy matching."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.utils.config import get as cfg_get
from src.utils.logger import get_logger

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = _REPO_ROOT / "config" / "skills_taxonomy.json"

# Common abbreviations -> canonical taxonomy entry. Applied on whole-word
# matches only (word boundaries) to avoid e.g. matching "ai" inside "email"
# or "ds" inside "days".
ABBREVIATION_MAP: dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "genai": "generative ai",
    "llm": "large language models",
    "llms": "large language models",
    "rl": "reinforcement learning",
    "torch": "pytorch",
    "k8s": "kubernetes",
    "ec2": "aws",
}

# Multi-word abbreviations, matched as a phrase rather than a single token
# (e.g. "Azure ML" is the named Microsoft product "Azure Machine Learning",
# not just a generic "azure" + "ml" co-occurrence).
PHRASE_ABBREVIATION_MAP: dict[str, str] = {
    "azure ml": "azure machine learning",
    "aws ec2": "aws",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.#-]*")


def _contains_skill(text_lower: str, skill: str) -> bool:
    """Substring match with non-alphanumeric boundaries on both sides.

    Plain `skill in text` is not safe for short/ambiguous taxonomy entries:
    "r" would match inside "experience" and "go" would match inside
    "django". Requiring both edges to be non-alphanumeric (or string start
    /end) avoids that while still matching entries containing punctuation
    like "c++", "c#", or "next.js".
    """
    start = 0
    while True:
        idx = text_lower.find(skill, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
        end = idx + len(skill)
        after_ok = end == len(text_lower) or not text_lower[end].isalnum()
        if before_ok and after_ok:
            return True
        start = idx + 1


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, list[str]]:
    path = Path(cfg_get("features.skills.taxonomy_path", str(TAXONOMY_PATH)))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _flat_skill_set() -> frozenset[str]:
    taxonomy = load_taxonomy()
    return frozenset(s.lower() for group in taxonomy.values() for s in group)


def _abbreviation_hits(text_lower: str) -> set[str]:
    """Find standalone abbreviation tokens (ML, DL, AI, K8s, EC2, ...) and
    map them to their canonical, full-form taxonomy entry.

    The token pattern allows digits (not just letters) so alphanumeric
    abbreviations like "k8s" and "ec2" are still recognized as whole words.
    """
    hits: set[str] = set()
    tokens = set(re.findall(r"\b[a-z][a-z0-9]{1,5}\b", text_lower))
    for abbr, canonical in ABBREVIATION_MAP.items():
        if abbr in tokens:
            hits.add(canonical)
    return hits


def _phrase_abbreviation_hits(text_lower: str) -> set[str]:
    """Find multi-word abbreviations ("Azure ML", "AWS EC2") that need
    phrase-level matching rather than single-token lookup."""
    return {
        canonical
        for phrase, canonical in PHRASE_ABBREVIATION_MAP.items()
        if _contains_skill(text_lower, phrase)
    }


def _fuzzy_hits(text_lower: str, known_skills: frozenset[str], threshold: float) -> set[str]:
    """Catch misspellings ("Djnago" ~ "django") via token-level fuzzy match.

    Degrades gracefully (returns no extra hits) if `rapidfuzz` isn't
    installed, rather than crashing the whole extraction pipeline.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        log.debug("rapidfuzz not installed; skipping fuzzy skill matching")
        return set()

    hits: set[str] = set()
    tokens = {t.lower() for t in _TOKEN_RE.findall(text_lower) if len(t) > 2}
    # Only fuzzy-match single-word taxonomy entries; multi-word entries
    # ("spring boot") are handled by the substring pass and are noisy to
    # fuzzy-match token-by-token.
    single_word_skills = [s for s in known_skills if " " not in s]
    score_cutoff = threshold * 100
    for token in tokens:
        if token in known_skills:
            continue  # already an exact hit, no need to fuzzy-check it
        for skill in single_word_skills:
            if abs(len(token) - len(skill)) > 2:
                continue
            if fuzz.ratio(token, skill) >= score_cutoff:
                hits.add(skill)
                break
    return hits


def extract_skills(text: str) -> list[str]:
    """Return a deduplicated, sorted list of canonical skills found in the text.

    Pipeline: exact/substring taxonomy match -> abbreviation normalization
    (ML -> machine learning) -> fuzzy match for misspellings, all merged and
    deduplicated against the same canonical vocabulary.
    """
    known_skills = _flat_skill_set()
    text_lower = text.lower()

    exact_hits = {s for s in known_skills if _contains_skill(text_lower, s)}
    abbrev_hits = _abbreviation_hits(text_lower)
    phrase_hits = _phrase_abbreviation_hits(text_lower)

    threshold = cfg_get("features.skills.fuzzy_threshold", 0.85)
    fuzzy_hits = _fuzzy_hits(text_lower, known_skills, threshold)

    return sorted(exact_hits | abbrev_hits | phrase_hits | fuzzy_hits)
