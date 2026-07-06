"""Integration tests for the FastAPI app, using TestClient (in-process, no server)."""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

SAMPLE_JD = "Backend Engineer. 3-5 years. Required skills: Python, FastAPI, PostgreSQL."


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_parse_endpoint_rejects_unsupported_extension():
    resp = client.post("/parse", files={"file": ("resume.xyz", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_parse_endpoint_returns_structured_profile():
    content = b"Jane Doe\njane.doe@example.com\nSkills: Python, FastAPI, PostgreSQL"
    resp = client.post("/parse", files={"file": ("resume.txt", content, "text/plain")})
    assert resp.status_code == 200
    profile = resp.json()
    assert "python" in profile["skills"]


def test_score_endpoint_returns_breakdown():
    content = b"5 years experience.\nSkills: Python, FastAPI, PostgreSQL"
    resp = client.post(
        "/score",
        files={"file": ("resume.txt", content, "text/plain")},
        data={"jd_text": SAMPLE_JD},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["score"]["total"] <= 1.0


def test_rank_endpoint_skips_bad_files_and_still_ranks_the_rest():
    good_a = ("strong.txt", b"5 years.\nSkills: Python, FastAPI, PostgreSQL", "text/plain")
    good_b = ("weak.txt", b"1 year.\nSkills: HTML", "text/plain")
    bad = ("broken.pdf", b"not a real pdf", "application/pdf")

    resp = client.post(
        "/rank",
        files=[("files", good_a), ("files", good_b), ("files", bad)],
        data={"jd_text": SAMPLE_JD},
    )
    assert resp.status_code == 200
    ranked = resp.json()
    assert len(ranked) == 2
    assert ranked[0]["score"]["total"] >= ranked[1]["score"]["total"]


def test_parse_endpoint_sanitizes_path_traversal_filename():
    # A crafted filename shouldn't be able to influence where the temp file
    # is written, or crash the request — it should just fail extension
    # validation cleanly, the same as any other unsupported/missing extension.
    content = b"Jane Doe\njane.doe@example.com\nSkills: Python"
    resp = client.post(
        "/parse",
        files={"file": ("../../../etc/passwd", content, "text/plain")},
    )
    assert resp.status_code == 400


def test_parse_endpoint_rejects_oversized_upload(monkeypatch):
    from src import api as api_module

    monkeypatch.setattr(api_module, "cfg_get", lambda key, default=None: 0.000001)
    resp = client.post("/parse", files={"file": ("resume.txt", b"x" * 10_000, "text/plain")})
    assert resp.status_code == 400
    assert "exceeds" in resp.json()["detail"]
