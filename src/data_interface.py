"""Shared, checksum-verified training/validation data interface for all models."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"ham": 0, "spam": 1}


def load_partition(name: str) -> list[dict[str, str]]:
    """Return normalized rows; deliberately refuses held-out test loading."""
    if name not in {"train", "validation"}:
        raise ValueError("Progress-phase model code may load only train or validation; test remains reserved.")
    path = ROOT / "data" / "processed" / f"{name}.csv"
    audit = json.loads((ROOT / "artifacts" / "data_audit.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != audit["splits"][name]["sha256"]:
        raise RuntimeError(f"{name} CSV checksum differs from audited split.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == audit["splits"][name]["count"]
    assert len({row["row_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["label"] in LABEL_TO_ID
        assert hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["row_id"]
    return rows


def texts_and_labels(name: str) -> tuple[list[str], list[int], list[str]]:
    rows = load_partition(name)
    return [row["text"] for row in rows], [LABEL_TO_ID[row["label"]] for row in rows], [row["row_id"] for row in rows]
