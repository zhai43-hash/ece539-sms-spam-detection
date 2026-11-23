"""Verify saved progress artifacts without predicting on or parsing test text."""
import csv
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.base import clone

from data_interface import ROOT, load_partition, texts_and_labels
from train_baseline import calculate_metrics


def main():
    artifacts = ROOT / "artifacts"
    audit = json.loads((artifacts / "data_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifacts / "split_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifacts / "baseline_metrics.json").read_text(encoding="utf-8"))
    checks = []
    for name, metadata in audit["dataset"]["raw_files"].items():
        assert hashlib.sha256((ROOT / "data" / "raw" / name).read_bytes()).hexdigest() == metadata["sha256"]
    checks.append("ZIP and extracted official corpus/readme match recorded SHA256 checksums")
    for name, metadata in audit["splits"].items():
        # Reserved test access is strictly a byte checksum, not model loading or evaluation.
        assert hashlib.sha256((ROOT / metadata["path"]).read_bytes()).hexdigest() == metadata["sha256"]
        assert manifest["partitions"][name]["sha256"] == metadata["sha256"]
        assert len(manifest["partitions"][name]["row_ids"]) == metadata["count"]
    checks.append("All partition file hashes and row-ID manifest counts match the initial audit")
    id_sets = {name: set(data["row_ids"]) for name, data in manifest["partitions"].items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert id_sets[a].isdisjoint(id_sets[b])
    checks.append("No shared row IDs between any pair of partitions")
    try:
        load_partition("test")
    except ValueError:
        pass
    else:
        raise AssertionError("The progress-phase data interface must reject test loading")
    checks.append("Shared model data interface rejects reserved test loading")
    x_train, _, _ = texts_and_labels("train")
    x_val, y_val_list, val_ids = texts_and_labels("validation")
    bundle = joblib.load(ROOT / metrics["artifacts"]["model_path"])
    pipeline = bundle["pipeline"]
    refit_vectorizer = clone(pipeline["tfidf"]).fit(x_train)
    assert refit_vectorizer.vocabulary_ == pipeline["tfidf"].vocabulary_
    np.testing.assert_array_equal(refit_vectorizer.idf_, pipeline["tfidf"].idf_)
    checks.append("Saved TF-IDF vocabulary and IDF exactly match a training-only refit")
    with (ROOT / metrics["artifacts"]["validation_predictions_path"]).open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert [row["row_id"] for row in saved_rows] == val_ids
    probability = pipeline.predict_proba(x_val)[:, 1]
    np.testing.assert_allclose(probability, [float(row["spam_probability"]) for row in saved_rows], rtol=0, atol=1e-14)
    recalculated = calculate_metrics(np.array(y_val_list), probability, bundle["threshold"])
    assert recalculated == metrics["validation_selected_threshold"]
    checks.append("Reloaded saved model reproduces validation probabilities and all selected-threshold metrics")
    output = {"verified_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "checks": checks, "test_status": "No test text parsed and no test predictions or performance metrics computed; byte-checksum bookkeeping only.",
              "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT / "src").glob("*.py"))}}
    (artifacts / "verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
