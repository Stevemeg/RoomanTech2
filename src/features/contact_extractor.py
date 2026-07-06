"""Extract contact information (name, email, phone, LinkedIn)."""

from __future__ import annotations

import re

from src.utils.config import get as cfg_get
from src.utils.logger import get_logger

log = get_logger(__name__)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[a-zA-Z0-9_-]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"github\.com/[a-zA-Z0-9_-]+", re.IGNORECASE)

# Lines that look like section headers, not a person's name.
_NON_NAME_WORDS = {
    "resume",
    "curriculum",
    "vitae",
    "cv",
    "profile",
    "summary",
    "objective",
    "contact",
    "portfolio",
    "education",
    "experience",
    "skills",
}

_SPACY_NLP = None
_SPACY_LOAD_FAILED = False


def _get_spacy_nlp():
    """Lazily load spaCy, caching the pipeline across calls.

    spaCy + its model are heavy, optional dependencies. If either isn't
    installed (e.g. offline / minimal environments), we fall back to a
    regex/heuristic name guess rather than crashing the whole pipeline.
    """
    global _SPACY_NLP, _SPACY_LOAD_FAILED
    if _SPACY_NLP is not None or _SPACY_LOAD_FAILED:
        return _SPACY_NLP
    try:
        import spacy

        model_name = cfg_get("nlp.spacy_model", "en_core_web_sm")
        _SPACY_NLP = spacy.load(model_name, disable=["parser", "lemmatizer"])
    except Exception as exc:  # noqa: BLE001
        log.debug(f"spaCy unavailable, using heuristic name extraction ({exc})")
        _SPACY_LOAD_FAILED = True
        _SPACY_NLP = None
    return _SPACY_NLP


def _looks_like_name(line: str) -> bool:
    line = line.strip()
    if not (2 <= len(line.split()) <= 4):
        return False
    if any(ch.isdigit() for ch in line):
        return False
    if EMAIL_RE.search(line) or PHONE_RE.search(line) or LINKEDIN_RE.search(line):
        return False
    lowered = line.lower()
    if any(w in lowered for w in _NON_NAME_WORDS):
        return False
    words = line.replace(",", "").split()
    # Title-case or fully-capitalised header names both count.
    return all(w[0].isupper() for w in words if w[0].isalpha())


def _heuristic_name(text: str) -> str | None:
    for line in text.splitlines()[:6]:
        if _looks_like_name(line):
            return line.strip()
    return None


def _spacy_name(text: str) -> str | None:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return None
    head = "\n".join(text.splitlines()[:8])
    doc = nlp(head)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return None


def extract_contact(text: str) -> dict:
    """Return {name, email, phone, linkedin, github, all_emails, all_phones}.

    Resumes sometimes list a personal + a university email, or a home +
    mobile number — we keep the first of each as the "primary" value for
    backward-compat callers, but expose the full list too.
    """
    emails = list(dict.fromkeys(m.group(0) for m in EMAIL_RE.finditer(text)))
    phones = list(dict.fromkeys(m.group(0).strip() for m in PHONE_RE.finditer(text)))
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)

    name = _spacy_name(text) or _heuristic_name(text)

    return {
        "name": name,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
        "all_emails": emails,
        "all_phones": phones,
    }
