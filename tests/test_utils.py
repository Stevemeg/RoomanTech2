"""Tests for src.utils.io and src.utils.config."""

from dataclasses import dataclass

from src.utils.config import get as cfg_get
from src.utils.config import load_config
from src.utils.io import load_json, read_text, save_csv, save_json


@dataclass
class _Dummy:
    a: int
    b: str


def test_save_and_load_json_roundtrip(tmp_path):
    path = tmp_path / "out.json"
    save_json({"x": 1, "y": [1, 2, 3]}, path)
    assert load_json(path) == {"x": 1, "y": [1, 2, 3]}


def test_save_json_accepts_dataclass(tmp_path):
    path = tmp_path / "out.json"
    save_json(_Dummy(a=1, b="hi"), path)
    assert load_json(path) == {"a": 1, "b": "hi"}


def test_read_text(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("hello world", encoding="utf-8")
    assert read_text(path) == "hello world"


def test_save_csv_flattens_nested_values(tmp_path):
    path = tmp_path / "out.csv"
    save_csv([{"rank": 1, "file": "a.pdf", "missing_skills": ["x", "y"]}], path)
    content = path.read_text(encoding="utf-8")
    assert "rank" in content.splitlines()[0]
    assert "a.pdf" in content


def test_save_csv_handles_empty_rows(tmp_path):
    path = tmp_path / "empty.csv"
    save_csv([], path)
    assert path.exists()


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "ranking" in cfg


def test_config_get_dotted_key_with_default():
    assert cfg_get("similarity.tfidf_weight") == 0.4
    assert cfg_get("does.not.exist", "fallback") == "fallback"


def test_env_var_overrides_config_value(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
    assert cfg_get("similarity.embedding_model") == "some/other-model"


def test_env_var_override_coerces_to_float(monkeypatch):
    monkeypatch.setenv("WEIGHT_SKILLS", "0.7")
    value = cfg_get("ranking.weights.skills", 0.45)
    assert isinstance(value, float)
    assert value == 0.7
