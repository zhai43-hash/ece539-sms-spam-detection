"""Download, audit, normalize, deduplicate, and partition the official UCI corpus.

No model is trained here. Partition construction is the only test-set access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
METADATA_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
EXPECTED_ZIP_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
SEED = 539
FIELDS = ["row_id", "label", "text", "source_line_ids", "duplicate_count"]
PARTITIONS = ("train", "validation", "test")
RATIOS = (0.70, 0.15, 0.15)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def allocate(n: int) -> list[int]:
    """Largest remainder allocation per class; equal remainders use split order."""
    exact = [n * ratio for ratio in RATIOS]
    counts = [math.floor(x) for x in exact]
    order = sorted(range(len(RATIOS)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the checksum-locked official ZIP if absent")
    args = parser.parse_args()
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    artifacts_dir = ROOT / "artifacts"
    for directory in (raw_dir, processed_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "sms_spam_collection.zip"
    if not zip_path.exists():
        if not args.download:
            raise SystemExit("Official ZIP is absent; run with --download.")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    actual_zip_sha = sha256_file(zip_path)
    if actual_zip_sha != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"ZIP checksum differs from locked corpus: {actual_zip_sha}")
    with ZipFile(zip_path) as archive:
        for name in ("SMSSpamCollection", "readme"):
            (raw_dir / name).write_bytes(archive.read(name))

    # A record is one physical LF-delimited line, split at its FIRST TAB.
    # Quotes inside SMS text are literal; CSV quote interpretation is inappropriate.
    decoded = (raw_dir / "SMSSpamCollection").read_bytes().decode("utf-8", errors="strict")
    lines = decoded.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    raw_label_counts: Counter = Counter()
    malformed_ids, empty_ids = [], []
    valid_rows = 0
    for line_id, line in enumerate(lines, 1):
        line = line.removesuffix("\r")
        if "\t" not in line:
            malformed_ids.append(line_id)
            continue
        label, text = line.split("\t", 1)
        if label not in {"ham", "spam"}:
            malformed_ids.append(line_id)
            continue
        raw_label_counts[label] += 1
        normalized = normalize(text)
        if not normalized:
            empty_ids.append(line_id)
            continue
        valid_rows += 1
        groups[normalized].append((line_id, label))

    clean_rows, conflicting_groups = [], []
    duplicate_removed = 0
    repeated_consistent_groups = 0
    for text, source in groups.items():
        labels = {label for _, label in source}
        row_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if len(labels) > 1:
            conflicting_groups.append({"row_id": row_id, "source_line_ids": [n for n, _ in source], "labels": sorted(labels)})
            continue
        duplicate_removed += len(source) - 1
        repeated_consistent_groups += int(len(source) > 1)
        clean_rows.append({"row_id": row_id, "label": source[0][1], "text": text,
                           "source_line_ids": json.dumps([n for n, _ in source], separators=(",", ":")),
                           "duplicate_count": len(source)})
    clean_rows.sort(key=lambda row: row["row_id"])
    assert len(clean_rows) == len({row["row_id"] for row in clean_rows})
    assert len(clean_rows) == len({row["text"] for row in clean_rows})
    conflict_row_count = sum(len(group["source_line_ids"]) for group in conflicting_groups)
    assert valid_rows == len(clean_rows) + duplicate_removed + conflict_row_count
    assert len(lines) == valid_rows + len(malformed_ids) + len(empty_ids)

    partitions: dict[str, list[dict]] = {name: [] for name in PARTITIONS}
    for label in ("ham", "spam"):
        class_rows = [row for row in clean_rows if row["label"] == label]
        random.Random(f"{SEED}:{label}").shuffle(class_rows)
        counts = allocate(len(class_rows))
        offset = 0
        for name, count in zip(PARTITIONS, counts):
            partitions[name].extend(class_rows[offset : offset + count])
            offset += count
        assert offset == len(class_rows)
    for rows in partitions.values():
        rows.sort(key=lambda row: row["row_id"])

    ids = {name: {row["row_id"] for row in rows} for name, rows in partitions.items()}
    texts = {name: {row["text"] for row in rows} for name, rows in partitions.items()}
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{a}_{b}"] = {"row_ids": len(ids[a] & ids[b]), "normalized_texts": len(texts[a] & texts[b])}
        assert not ids[a] & ids[b]
        assert not texts[a] & texts[b]
    assert set.union(*ids.values()) == {row["row_id"] for row in clean_rows}

    manifest = {"seed": SEED, "partitions": {}}
    split_stats = {}
    for name, rows in partitions.items():
        path = processed_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        split_stats[name] = {"count": len(rows), "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
                             "fraction": len(rows) / len(clean_rows), "sha256": sha256_file(path),
                             "path": str(path.relative_to(ROOT)).replace("\\", "/")}
        manifest["partitions"][name] = {"sha256": sha256_file(path), "row_ids": [row["row_id"] for row in rows]}
    write_json(artifacts_dir / "split_manifest.json", manifest)
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": "UCI SMS Spam Collection", "download_url": DATA_URL,
                    "metadata_url": METADATA_URL, "doi": "10.24432/C5CC84", "license": "CC BY 4.0",
                    "catalog_instance_count": 5574,
                    "raw_files": {name: {"sha256": sha256_file(raw_dir / name), "bytes": (raw_dir / name).stat().st_size}
                                  for name in ("sms_spam_collection.zip", "SMSSpamCollection", "readme")}},
        "parser": "UTF-8 strict; physical LF-delimited records; split at first TAB; ham/spam labels only; quotes literal",
        "normalization": "Python str.lower(), then whitespace split/join with one ASCII space",
        "deduplication": "Group by normalized text; collapse consistent-label groups; exclude ALL rows of conflicting-label groups",
        "counts": {"raw_physical_lines": len(lines), "raw_label_counts": dict(sorted(raw_label_counts.items())),
                   "malformed_excluded": len(malformed_ids), "empty_excluded": len(empty_ids),
                   "valid_nonempty_rows": valid_rows, "consistent_duplicate_groups": repeated_consistent_groups,
                   "consistent_duplicate_rows_removed": duplicate_removed,
                   "conflicting_groups_excluded": len(conflicting_groups), "conflicting_rows_excluded": conflict_row_count,
                   "clean_unique_rows": len(clean_rows),
                   "clean_label_counts": dict(sorted(Counter(row["label"] for row in clean_rows).items()))},
        "excluded": {"malformed_line_ids": malformed_ids, "empty_line_ids": empty_ids,
                     "conflicting_groups": conflicting_groups},
        "split_method": "Per-class SHA256-sorted rows, random.Random(str(seed)+':'+label).shuffle; largest-remainder allocation; ties train, validation, test; output SHA256-sorted",
        "seed": SEED, "target_ratios": dict(zip(PARTITIONS, RATIOS)), "splits": split_stats,
        "overlap_counts": overlaps,
        "assertions_passed": ["valid-row accounting", "raw-row accounting", "unique IDs and normalized texts", "no pairwise split overlap", "all clean IDs assigned exactly once"],
        "test_policy": "Reserved. No test predictions, threshold tuning, model selection, or test performance metrics during progress phase.",
        "environment": {"python": platform.python_version()},
    }
    write_json(artifacts_dir / "data_audit.json", audit)
    print(json.dumps({"counts": audit["counts"], "splits": split_stats, "overlap_counts": overlaps}, indent=2))


if __name__ == "__main__":
    main()
