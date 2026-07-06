from dataclasses import dataclass, field

from src.features.certifications_extractor import extract_certifications
from src.features.contact_extractor import extract_contact
from src.features.education_extractor import extract_education
from src.features.experience_extractor import compute_total_experience_years, extract_experience
from src.features.skills_extractor import extract_skills


@dataclass
class CandidateProfile:
    raw_text: str
    contact: dict = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    total_experience_years: float = 0.0


def extract_features(text: str) -> CandidateProfile:
    """Run all feature extractors and return a structured profile."""
    experience = extract_experience(text)
    return CandidateProfile(
        raw_text=text,
        contact=extract_contact(text),
        skills=extract_skills(text),
        experience=experience,
        education=extract_education(text),
        certifications=extract_certifications(text),
        total_experience_years=compute_total_experience_years(experience, text),
    )


__all__ = ["CandidateProfile", "extract_features"]
