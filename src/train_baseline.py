"""Fit one logistic-regression baseline; choose threshold on validation only."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from data_interface import ROOT, texts_and_labels

SEED = 539


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "n": len(y_true),
            "spam_precision": float(precision_score(y_true, predicted, zero_division=0)),
            "spam_recall": float(recall_score(y_true, predicted, zero_division=0)),
            "spam_f1": float(f1_score(y_true, predicted, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predicted)),
            "ham_false_positive_rate": float(fp / (fp + tn)),
            "confusion_matrix": {"true_ham_pred_ham_tn": int(tn), "true_ham_pred_spam_fp": int(fp),
                                 "true_spam_pred_ham_fn": int(fn), "true_spam_pred_spam_tp": int(tp)}}


def main() -> None:
    x_train, y_train_list, train_ids = texts_and_labels("train")
    x_val, y_val_list, val_ids = texts_and_labels("validation")
    assert set(train_ids).isdisjoint(val_ids)
    y_train, y_val = np.array(y_train_list), np.array(y_val_list)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=20000,
                                  min_df=1, sublinear_tf=True)),
        ("logistic", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                         max_iter=2000, random_state=SEED)),
    ])
    started = datetime.now(timezone.utc)
    pipeline.fit(x_train, y_train)
    assert list(pipeline.classes_) == [0, 1]
    val_probability = pipeline.predict_proba(x_val)[:, 1]
    # Candidate values cover every distinct validation prediction pattern.
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], val_probability,
                                           [np.nextafter(val_probability.max(), np.inf)])))
    candidate_metrics = [calculate_metrics(y_val, val_probability, threshold) for threshold in thresholds]
    selected = max(candidate_metrics, key=lambda m: (m["spam_f1"], -m["ham_false_positive_rate"], m["threshold"]))
    default_metrics = calculate_metrics(y_val, val_probability, 0.5)
    model_path = ROOT / "models" / "tfidf_logistic.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": selected["threshold"], "label_to_id": {"ham": 0, "spam": 1}}, model_path)
    artifacts = ROOT / "artifacts"
    pred_path = artifacts / "baseline_validation_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_id", "label", "spam_probability", "prediction_selected_threshold"])
        for row_id, label, probability in zip(val_ids, y_val, val_probability):
            writer.writerow([row_id, int(label), format(float(probability), ".17g"), int(probability >= selected["threshold"])])
    audit = json.loads((artifacts / "data_audit.json").read_text(encoding="utf-8"))
    result = {
        "experiment": "initial TF-IDF + logistic regression baseline", "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "training_count": len(y_train), "validation_count": len(y_val),
        "dataset_zip_sha256": audit["dataset"]["raw_files"]["sms_spam_collection.zip"]["sha256"],
        "training_csv_sha256": audit["splits"]["train"]["sha256"],
        "validation_csv_sha256": audit["splits"]["validation"]["sha256"],
        "model": {"tfidf": {"ngram_range": [1, 2], "max_features": 20000, "min_df": 1,
                             "sublinear_tf": True, "lowercase": False, "norm": "l2",
                             "token_pattern": pipeline["tfidf"].token_pattern,
                             "fitted_vocabulary_size": len(pipeline["tfidf"].vocabulary_),
                             "fit_scope": "training messages only"},
                  "logistic_regression": {"C": 1.0, "class_weight": "balanced", "solver": "liblinear",
                                          "max_iter": 2000, "random_state": SEED,
                                          "iterations": pipeline["logistic"].n_iter_.tolist()}},
        "threshold_selection": {"partition": "validation", "candidate_count": len(thresholds),
                                "candidate_rule": "all unique validation probabilities plus 0, 0.5, 1 and nextafter(max_probability,+infinity)",
                                "prediction_rule": "spam if probability >= threshold",
                                "objective": "maximum validation spam F1", "tie_break": "lower ham false-positive rate, then higher threshold",
                                "selected_threshold": selected["threshold"]},
        "validation_selected_threshold": selected, "validation_default_0_5": default_metrics,
        "test_status": "NOT EVALUATED. Model code never loads test.csv; no test predictions or numeric test metrics exist.",
        "limitations": ["Single data split and initial fixed hyperparameter configuration",
                        "Selected-threshold validation metrics also informed threshold choice, so they are optimistic development results",
                        "Historical SMS corpus may not represent current spam or deployment populations",
                        "No final generalization claim before the frozen protocol is evaluated once on held-out test"],
        "artifacts": {"model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                      "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                      "validation_predictions_path": str(pred_path.relative_to(ROOT)).replace("\\", "/"),
                      "validation_predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest()},
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (artifacts / "baseline_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_selected_threshold": selected, "validation_default_0_5": default_metrics,
                      "test_status": result["test_status"]}, indent=2))


if __name__ == "__main__":
    main()
