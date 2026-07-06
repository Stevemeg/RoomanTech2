"""Tests for resume parsers."""

import pytest

from src.parser import parse_resume
from src.parser.text_cleaner import clean_text
from src.utils.exceptions import (
    CorruptedDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    UnsupportedFileTypeError,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    world") == "hello world"


def test_clean_text_normalizes_bullets():
    assert "-" in clean_text("• item one")


def test_clean_text_fixes_hyphenated_linebreaks():
    assert clean_text("develop-\nment") == "development"


def test_clean_text_strips_page_numbers():
    cleaned = clean_text("Line one\nPage 3\nLine two")
    assert "Page 3" not in cleaned
    assert "Line one" in cleaned and "Line two" in cleaned


@pytest.fixture
def sample_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Jane Doe")
    page.insert_text((72, 90), "jane.doe@example.com")
    page.insert_text((72, 108), "Skills: Python, Django")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("John Smith")
    document.add_paragraph("john.smith@example.com")
    document.save(str(path))
    return path


def test_parse_resume_reads_pdf(sample_pdf):
    text = parse_resume(str(sample_pdf))
    assert "Jane Doe" in text
    assert "jane.doe@example.com" in text


def test_parse_resume_reads_docx(sample_docx):
    text = parse_resume(str(sample_docx))
    assert "John Smith" in text


def test_parse_resume_reads_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Plain text resume content", encoding="utf-8")
    assert "Plain text resume content" in parse_resume(str(path))


def test_parse_resume_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "resume.xyz"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        parse_resume(str(path))


def test_parse_resume_rejects_empty_txt(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        parse_resume(str(path))


def test_parse_resume_handles_corrupted_pdf(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a real pdf")
    with pytest.raises(CorruptedDocumentError):
        parse_resume(str(path))


def test_parse_resume_handles_encrypted_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "encrypted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret resume")
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret123",
    )
    doc.close()
    with pytest.raises(EncryptedDocumentError):
        parse_resume(str(path))


def test_parse_resume_handles_image_only_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "image_only.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(str(path))
    doc.close()
    with pytest.raises(EmptyDocumentError):
        parse_resume(str(path))


def test_parse_resume_rejects_oversized_file(tmp_path, monkeypatch):
    from src.utils.exceptions import FileTooLargeError

    def fake_cfg_get(key, default=None):
        return 0 if "max_file_size_mb" in key else default

    monkeypatch.setattr("src.parser.cfg_get", fake_cfg_get)

    path = tmp_path / "big.txt"
    path.write_text("x" * 1000, encoding="utf-8")
    with pytest.raises(FileTooLargeError):
        parse_resume(str(path))
