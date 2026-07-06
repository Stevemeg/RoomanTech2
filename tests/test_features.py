"""Tests for feature extractors."""

from src.features import extract_features
from src.features.certifications_extractor import extract_certifications
from src.features.contact_extractor import extract_contact
from src.features.education_extractor import extract_education, highest_degree_level
from src.features.experience_extractor import compute_total_experience_years, extract_experience
from src.features.skills_extractor import extract_skills


def test_extract_contact_finds_email():
    text = "Reach me at jane.doe@example.com for opportunities."
    assert extract_contact(text)["email"] == "jane.doe@example.com"


def test_extract_contact_finds_multiple_emails_and_phones():
    text = "jane@example.com or backup@school.edu | 9876543210 / 8123456780"
    contact = extract_contact(text)
    assert len(contact["all_emails"]) == 2
    assert len(contact["all_phones"]) == 2


def test_extract_skills_finds_python():
    text = "5 years of Python and Django experience with PostgreSQL."
    skills = extract_skills(text)
    assert "python" in skills
    assert "django" in skills
    assert "postgresql" in skills


def test_extract_skills_normalizes_abbreviations():
    skills = extract_skills("Background in ML, DL and NLP.")
    assert "machine learning" in skills
    assert "deep learning" in skills
    assert "natural language processing" in skills


def test_extract_skills_normalizes_infra_abbreviations():
    skills = extract_skills("Deployed with K8s on AWS EC2, trained models in Torch.")
    assert "kubernetes" in skills
    assert "aws" in skills
    assert "pytorch" in skills


def test_extract_skills_normalizes_azure_ml_phrase():
    skills = extract_skills("Built pipelines with Azure ML for training.")
    assert "azure machine learning" in skills


def test_extract_skills_avoids_short_token_false_positives():
    # "r" (the language) must not match as a substring of "experience".
    skills = extract_skills("I have a lot of experience in backend development.")
    assert "r" not in skills


def test_extract_experience_parses_date_range_and_role_company():
    text = "Experience\nSenior Backend Engineer at Acme Corp\nJan 2020 - Present\n"
    entries = extract_experience(text)
    assert len(entries) == 1
    assert entries[0]["company"] == "Acme Corp"
    assert entries[0]["role"] == "Senior Backend Engineer"
    assert entries[0]["duration_years"] > 0


def test_extract_experience_detects_employment_type():
    text = (
        "Experience\n"
        "Data Science Intern at Acme Corp (Internship)\n"
        "Jun 2023 - Aug 2023\n\n"
        "Senior Engineer at Beta Inc\n"
        "Jan 2020 - Present\n"
        "- Full-time role leading the platform team\n"
    )
    entries = extract_experience(text)
    assert entries[0]["employment_type"] == "Internship"
    assert entries[1]["employment_type"] == "Full-time"


def test_extract_experience_employment_type_defaults_to_none_when_unstated():
    text = "Experience\nBackend Developer at Beta Inc\nJan 2020 - Jan 2022\n"
    entries = extract_experience(text)
    assert entries[0]["employment_type"] is None


def test_compute_total_experience_falls_back_to_summary_line():
    text = "3+ years of experience building web applications."
    assert compute_total_experience_years([], text) == 3.0


def test_extract_education_parses_degree_institution_field_year():
    text = "Education\nB.Tech in Computer Science, Acharya Institute of Technology, 2018 - 2022"
    entries = extract_education(text)
    assert entries[0]["degree"].lower().replace(".", "") == "btech"
    assert entries[0]["institution"] == "Acharya Institute of Technology"
    assert entries[0]["field"] == "Computer Science"
    assert entries[0]["year"] == "2022"


def test_highest_degree_level_ranks_masters_above_bachelors():
    bachelor = [{"degree": "Bachelor"}]
    master = [{"degree": "Master"}]
    assert highest_degree_level(master) > highest_degree_level(bachelor)


def test_extract_certifications_from_dedicated_section():
    text = "Certifications\nAWS Certified Solutions Architect\nGoogle Data Analytics Certificate"
    certs = extract_certifications(text)
    assert len(certs) == 2


def test_extract_features_populates_total_experience_years():
    text = "Experience\nBackend Developer at Beta Inc\nJan 2020 - Jan 2022\n"
    profile = extract_features(text)
    assert profile.total_experience_years > 0
