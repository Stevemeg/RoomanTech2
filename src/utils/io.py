"""File I/O helpers."""

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str | Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def save_csv(rows: list[dict], path: str | Path) -> None:
    """Write a flat list of dicts to CSV. Nested lists/dicts in any column
    are serialized to a JSON string so the CSV stays one-row-per-candidate
    (spreadsheet-friendly) instead of exploding into multiple rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat_row = {
                k: (json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
                for k, v in row.items()
            }
            writer.writerow(flat_row)
