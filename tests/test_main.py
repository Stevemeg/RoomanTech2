"""Tests for src.main: duplicate detection and safe output writing."""

from src.features import CandidateProfile
from src.main import _warn_on_duplicates, _write_csv_output, _write_output


def _profile(email: str, text: str) -> CandidateProfile:
    return CandidateProfile(raw_text=text, contact={"email": email})


def test_warn_on_duplicates_detects_same_email():
    profiles = [
        _profile("a@example.com", "Resume A"),
        _profile("a@example.com", "Resume A (resubmitted)"),
    ]
    messages = _warn_on_duplicates(["a.pdf", "a_v2.pdf"], profiles)
    assert any("Duplicate candidate email" in msg for msg in messages)


def test_warn_on_duplicates_detects_identical_text():
    profiles = [
        _profile("a@example.com", "Same resume text"),
        _profile("b@example.com", "Same resume text"),
    ]
    messages = _warn_on_duplicates(["a.pdf", "b.pdf"], profiles)
    assert any("Duplicate resume content" in msg for msg in messages)


def test_warn_on_duplicates_silent_for_distinct_candidates():
    profiles = [
        _profile("a@example.com", "Resume A"),
        _profile("b@example.com", "Resume B"),
    ]
    messages = _warn_on_duplicates(["a.pdf", "b.pdf"], profiles)
    assert messages == []


def test_write_output_handles_permission_error(tmp_path, monkeypatch):
    from src import main as main_module

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(main_module, "save_json", raise_permission_error)
    assert _write_output({"x": 1}, str(tmp_path / "out.json")) is False


def test_write_csv_output_handles_permission_error(tmp_path, monkeypatch):
    from src import main as main_module

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(main_module, "save_csv", raise_permission_error)
    assert _write_csv_output([{"a": 1}], str(tmp_path / "out.csv")) is False


def test_write_output_succeeds_normally(tmp_path):
    path = tmp_path / "out.json"
    assert _write_output({"x": 1}, str(path)) is True
    assert path.exists()


def test_main_exits_cleanly_on_missing_resumes_dir(tmp_path, capsys):
    import sys

    from src.main import main

    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Backend Engineer", encoding="utf-8")

    argv = [
        "main.py",
        "--resumes-dir",
        str(tmp_path / "does_not_exist"),
        "--jd",
        str(jd_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit for a missing resumes dir")
    finally:
        sys.argv = old_argv

    assert "does not exist" in capsys.readouterr().err


def test_main_exits_cleanly_on_missing_jd_file(tmp_path, capsys):
    import sys

    from src.main import main

    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Python developer", encoding="utf-8")

    argv = [
        "main.py",
        "--resume",
        str(resume_path),
        "--jd",
        str(tmp_path / "does_not_exist.txt"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit for a missing JD file")
    finally:
        sys.argv = old_argv

    assert "Could not read JD file" in capsys.readouterr().err
