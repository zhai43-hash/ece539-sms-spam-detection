"""Run one reproducible, CPU-only preliminary SMS CNN experiment.

Only train.csv and validation.csv are opened. The held-out test set is deliberately
not an accepted input. This is a pilot, not a multi-seed or ablation experiment.

Run from the project directory: python train_cnn.py
An existing result is protected unless --overwrite is supplied. An explicit
repeat requires the same configuration/data and verifies identical validation
probabilities and fixed-threshold metrics before replacing report artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "cnn_pilot"
MODEL_OUT = ROOT / "models" / "cnn_pilot"
CONFIG = {
    "experiment": "single_seed_preliminary_cpu_cnn_validation_only",
    "seed": 539,
    "train_file": "data/processed/train.csv",
    "validation_file": "data/processed/validation.csv",
    "allowed_partitions": ["train", "validation"],
    "test_evaluated": False,
    "tokenizer": r"\b\w+\b",
    "preprocessing": "input text is lowercase and whitespace-normalized",
    "vocab_fit_partition": "train",
    "max_vocab": 10000,
    "min_token_frequency": 2,
    "max_length": 128,
    "embedding_dim": 64,
    "conv_channels": 64,
    "kernel_size": 3,
    "conv_padding": "same; pool only positions centered on non-padding tokens",
    "dropout": 0.3,
    "batch_size": 64,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "loss": "BCEWithLogitsLoss (unweighted)",
    "max_epochs": 12,
    "early_stopping_patience": 3,
    "selection_metric": "minimum validation BCE loss",
    "fixed_threshold": 0.5,
    "threshold_grid": [round(x / 100, 2) for x in range(5, 100, 5)],
    "threshold_selection": "maximum validation F1; ties prefer closest to 0.5, then larger threshold",
    "device": "cpu",
    "cpu_threads": 2,
    "pretrained_embeddings": False,
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_allowed_partition(partition: str):
    if partition not in CONFIG["allowed_partitions"]:
        raise ValueError("This validation-only pilot forbids loading the test partition.")
    path = ROOT / "data" / "processed" / f"{partition}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["label"] in {"ham", "spam"} for row in rows)
    assert len({row["row_id"] for row in rows}) == len(rows)
    return rows, path


def tokenize(text: str):
    return re.findall(CONFIG["tokenizer"], text)


def build_vocab(rows):
    counts = Counter(token for row in rows for token in tokenize(row["text"]))
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in ordered:
        if counts[token] >= CONFIG["min_token_frequency"] and len(vocab) < CONFIG["max_vocab"]:
            vocab[token] = len(vocab)
    return vocab


def encode(rows, vocab):
    array = np.zeros((len(rows), CONFIG["max_length"]), dtype=np.int64)
    token_count = unknown_count = truncated_count = 0
    for index, row in enumerate(rows):
        tokens = tokenize(row["text"])
        truncated_count += int(len(tokens) > CONFIG["max_length"])
        tokens = tokens[: CONFIG["max_length"]]
        ids = [vocab.get(token, 1) for token in tokens]
        array[index, :len(ids)] = ids
        token_count += len(ids)
        unknown_count += sum(token == 1 for token in ids)
    labels = np.array([int(row["label"] == "spam") for row in rows], dtype=np.float32)
    stats = {
        "rows": len(rows), "spam_rows": int(labels.sum()),
        "ham_rows": int(len(labels) - labels.sum()),
        "empty_after_tokenization": int(np.sum(np.sum(array != 0, axis=1) == 0)),
        "rows_truncated": truncated_count,
        "retained_token_count": token_count,
        "unknown_token_count": unknown_count,
        "unknown_fraction": unknown_count / max(1, token_count),
    }
    return torch.from_numpy(array), torch.from_numpy(labels), stats


def masked_max_pool(features: torch.Tensor, valid_tokens: torch.Tensor):
    """Ignore every padding-centered position, even when convolution bias is positive.

    Features have shape [batch, channels, length]. Same-padding convolution can
    produce nonzero features at padded positions. Such positions cannot win the
    maximum. Empty sequences use an all-zero pooled vector instead of -infinity.
    """
    masked = features.masked_fill(~valid_tokens.unsqueeze(1), float("-inf"))
    maxima = masked.amax(dim=2)
    return torch.where(valid_tokens.any(dim=1, keepdim=True), maxima, torch.zeros_like(maxima))


class SpamCNN(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CONFIG["embedding_dim"], padding_idx=0)
        self.conv = nn.Conv1d(CONFIG["embedding_dim"], CONFIG["conv_channels"],
                              CONFIG["kernel_size"], padding=CONFIG["kernel_size"] // 2)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.output = nn.Linear(CONFIG["conv_channels"], 1)

    def forward(self, tokens):
        features = torch.relu(self.conv(self.embedding(tokens).transpose(1, 2)))
        pooled = masked_max_pool(features, tokens.ne(0))
        return self.output(self.dropout(pooled)).squeeze(1)


def load_pilot_checkpoint(model_directory: Path | None = None):
    """Return (CPU eval-mode model, vocabulary, checkpoint metadata).

    Example, with this module available from the project directory::

        model, vocabulary, metadata = load_pilot_checkpoint()
        rows, _ = load_allowed_partition("validation")
        tokens, labels, stats = encode(rows, vocabulary)
        with torch.no_grad():
            spam_probabilities = model(tokens).sigmoid()

    ``encode`` expects already lowercase, whitespace-normalized text and a
    ham/spam label. Checkpoint metadata contains its frozen configuration,
    vocabulary size, selected epoch, and validation loss. No optimizer state is
    stored; this checkpoint is for inference, not resuming interrupted training.
    """
    directory = Path(model_directory) if model_directory is not None else MODEL_OUT
    checkpoint = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != CONFIG:
        raise ValueError("Checkpoint configuration differs from this frozen pilot implementation.")
    vocabulary = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    assert len(vocabulary) == checkpoint["vocab_size"]
    model = SpamCNN(len(vocabulary))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model, vocabulary, metadata


def verify_invariants():
    # Padded features deliberately dwarf the valid values and must never win.
    features = torch.tensor([[[2., 3., 999., 999.], [-2., -1., 999., 999.]],
                             [[8., 8., 8., 8.], [9., 9., 9., 9.]]])
    validity = torch.tensor([[True, True, False, False], [False, False, False, False]])
    pooled = masked_max_pool(features, validity)
    assert torch.equal(pooled, torch.tensor([[3., -1.], [0., 0.]]))
    assert torch.isfinite(pooled).all()
    tiny_model = SpamCNN(10).eval()
    with torch.no_grad():
        short = tiny_model(torch.tensor([[2, 3, 4, 0]]))
        longer_padding = tiny_model(torch.tensor([[2, 3, 4, 0, 0, 0, 0]]))
        assert torch.allclose(short, longer_padding, atol=1e-6)
        assert torch.isfinite(tiny_model(torch.zeros((2, 7), dtype=torch.long))).all()
    train_only = build_vocab([{"text": "known known known"}])
    assert "heldout" not in train_only and "known" in train_only
    try:
        load_allowed_partition("test")
    except ValueError:
        pass
    else:
        raise AssertionError("Test partition must be rejected before any file access.")
    return {
        "padded_positions_cannot_win_pooling": True,
        "all_padding_pool_is_zero_and_logits_finite": True,
        "adding_padding_preserves_prediction": True,
        "vocabulary_fitted_only_on_supplied_training_rows": True,
        "test_partition_rejected_before_file_access": True,
    }


def compute_metrics(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "spam_precision": float(precision_score(labels, predicted, zero_division=0)),
        "spam_recall": float(recall_score(labels, predicted, zero_division=0)),
        "spam_f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_order": "rows=true ham/spam; columns=predicted ham/spam",
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    probabilities, labels_all = [], []
    loss_sum, count = 0.0, 0
    for tokens, labels in loader:
        logits = model(tokens)
        loss_sum += criterion(logits, labels).item() * len(labels)
        count += len(labels)
        probabilities.append(logits.sigmoid().numpy())
        labels_all.append(labels.numpy())
    return loss_sum / count, np.concatenate(labels_all).astype(int), np.concatenate(probabilities)


def main(overwrite: bool = False):
    """Train the frozen pilot; explicitly opt in to verified repetition with overwrite=True."""
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    previous_result = previous_predictions = None
    if (OUT / "metrics.json").exists():
        if not overwrite:
            raise RuntimeError("Pilot already completed. Use --overwrite (or main(overwrite=True)) "
                               "to explicitly repeat and verify this same frozen experiment.")
        previous_result = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
        prior_config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
        prior_config.pop("frozen_at_utc", None)
        if prior_config != CONFIG:
            raise ValueError("--overwrite only repeats the same frozen configuration; configuration changed.")
        for partition in CONFIG["allowed_partitions"]:
            path = ROOT / "data" / "processed" / f"{partition}.csv"
            if sha256(path) != previous_result["input_sha256"][partition]:
                raise ValueError(f"--overwrite requires unchanged {partition} data.")
        with (OUT / "validation_predictions.csv").open(encoding="utf-8", newline="") as handle:
            previous_predictions = list(csv.DictReader(handle))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    invariant_checks = verify_invariants()
    # Re-seed after invariant checks so they do not consume the training RNG.
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    write_json(OUT / "config.json", {**CONFIG, "frozen_at_utc": started_at})
    write_json(OUT / "invariant_checks.json", invariant_checks)
    train_rows, train_path = load_allowed_partition("train")
    validation_rows, validation_path = load_allowed_partition("validation")
    assert set(row["row_id"] for row in train_rows).isdisjoint(
        row["row_id"] for row in validation_rows)
    vocab = build_vocab(train_rows)
    train_x, train_y, train_stats = encode(train_rows, vocab)
    validation_x, validation_y, validation_stats = encode(validation_rows, vocab)
    assert train_x.max() < len(vocab) and validation_x.max() < len(vocab)
    write_json(MODEL_OUT / "vocab.json", vocab)
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=CONFIG["batch_size"],
                              shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(TensorDataset(validation_x, validation_y),
                                   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    model = SpamCNN(len(vocab))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    best_loss, best_epoch, patience = float("inf"), 0, 0
    history = []
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for tokens, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            assert torch.isfinite(loss), "Training loss became non-finite"
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        val_loss, labels, probabilities = evaluate(model, validation_loader, criterion)
        metrics = compute_metrics(labels, probabilities, CONFIG["fixed_threshold"])
        history.append({
            "epoch": epoch, "training_loss": total / count,
            "validation_loss": val_loss, "validation_spam_f1_at_0.5": metrics["spam_f1"],
            "validation_spam_precision_at_0.5": metrics["spam_precision"],
            "validation_spam_recall_at_0.5": metrics["spam_recall"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, patience = val_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": CONFIG,
                        "vocab_size": len(vocab), "best_epoch": best_epoch,
                        "validation_loss": best_loss}, MODEL_OUT / "best_checkpoint.pt")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                break
    checkpoint = torch.load(MODEL_OUT / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_loss, labels, probabilities = evaluate(model, validation_loader, criterion)
    fixed = compute_metrics(labels, probabilities, CONFIG["fixed_threshold"])
    threshold_grid_metrics = [compute_metrics(labels, probabilities, t) for t in CONFIG["threshold_grid"]]
    chosen = max(threshold_grid_metrics,
                 key=lambda m: (m["spam_f1"], -abs(m["threshold"] - 0.5), m["threshold"]))
    reproduction_check = None
    if previous_result is not None:
        same_rows = [row["row_id"] for row in validation_rows] == [row["row_id"] for row in previous_predictions]
        previous_probabilities = np.array([float(row["spam_probability"]) for row in previous_predictions])
        same_probabilities = np.array_equal(probabilities.astype(np.float64), previous_probabilities)
        same_metrics = fixed == previous_result["validation_fixed_threshold"]
        same_predictions = np.array_equal((probabilities >= 0.5).astype(int),
                                           [int(row["predicted_at_0.5"]) for row in previous_predictions])
        if not (same_rows and same_probabilities and same_metrics and same_predictions):
            raise RuntimeError("Deterministic repetition changed saved validation outcomes; "
                               "report metrics and prediction files have not been replaced.")
        reproduction_check = {
            "reason": "Explicit --overwrite verification after adding reproducible rerun support; no retuning.",
            "configuration_unchanged": True,
            "input_data_unchanged": True,
            "validation_row_order_identical": same_rows,
            "validation_probabilities_identical": same_probabilities,
            "fixed_threshold_predictions_identical": same_predictions,
            "fixed_threshold_metrics_identical": same_metrics,
            "previous_started_at_utc": previous_result["started_at_utc"],
            "previous_finished_at_utc": previous_result["finished_at_utc"],
            "previous_script_sha256": previous_result["script_sha256"],
            "completed_execution_count": previous_result.get("completed_execution_count", 1) + 1,
        }
    write_json(OUT / "threshold_grid.json", threshold_grid_metrics)
    with (OUT / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (OUT / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "true_label", "spam_probability",
                                                   "predicted_at_0.5", "predicted_at_selected_threshold"])
        writer.writeheader()
        for row, label, probability in zip(validation_rows, labels, probabilities):
            writer.writerow({"row_id": row["row_id"], "true_label": int(label),
                             "spam_probability": float(probability),
                             "predicted_at_0.5": int(probability >= 0.5),
                             "predicted_at_selected_threshold": int(probability >= chosen["threshold"])})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["training_loss"] for row in history], "o-", label="Training")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], "o-", label="Validation")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="Selected epoch")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [row["validation_spam_f1_at_0.5"] for row in history], "o-", color="#197278")
    axes[1].set(xlabel="Epoch", ylabel="Spam F1", title="Validation F1 at threshold 0.5", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xticks(epochs)
    fig.suptitle("Single-seed preliminary CNN pilot (seed 539)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=180)
    plt.close(fig)
    result = {
        "status": "completed",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "partitions_read": ["train", "validation"],
        "test_evaluated": False,
        "input_sha256": {"train": sha256(train_path), "validation": sha256(validation_path)},
        "script_sha256": sha256(Path(__file__)),
        "completed_execution_count": reproduction_check["completed_execution_count"] if reproduction_check else 1,
        "reproduction_check": reproduction_check,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "sklearn": sklearn.__version__,
                         "platform": platform.platform()},
        "seed": CONFIG["seed"],
        "vocabulary_size_including_special_tokens": len(vocab),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "training_data": train_stats,
        "validation_data": validation_stats,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": val_loss,
        "early_stopped": len(history) < CONFIG["max_epochs"],
        "validation_fixed_threshold": fixed,
        "validation_selected_threshold": chosen,
        "interpretation": (
            "Preliminary single-seed validation results, not final test results. "
            "The checkpoint and reported selected threshold both use validation data; "
            "selected-threshold F1 is tuning-optimistic. Fixed-threshold metrics still "
            "use a validation-selected checkpoint. No multi-seed estimate or ablation is complete."
        ),
        "invariant_checks": invariant_checks,
    }
    write_json(OUT / "metrics.json", result)
    write_json(OUT / "checkpoint_metadata.json", {
        "checkpoint": "models/cnn_pilot/best_checkpoint.pt",
        "checkpoint_sha256": sha256(MODEL_OUT / "best_checkpoint.pt"),
        "vocabulary": "models/cnn_pilot/vocab.json",
        "vocabulary_sha256": sha256(MODEL_OUT / "vocab.json"),
        "loader": "train_cnn.load_pilot_checkpoint() -> (model, vocabulary, metadata)",
        "device": "cpu",
        "model_mode_on_load": "eval",
        "stored_keys": ["state_dict", "config", "vocab_size", "best_epoch", "validation_loss"],
        "optimizer_state_stored": False,
        "best_epoch": best_epoch,
        "validation_loss": val_loss,
        "model_parameter_count": result["model_parameter_count"],
        "vocabulary_size_including_special_tokens": len(vocab),
        "special_tokens": {"<PAD>": 0, "<UNK>": 1},
        "maximum_sequence_length": CONFIG["max_length"],
        "prediction_rule": "sigmoid(logit) >= threshold; label 0=ham, 1=spam",
        "fixed_threshold": 0.5,
        "validation_selected_threshold": chosen["threshold"],
        "script_sha256": result["script_sha256"],
    })
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="explicitly repeat the same configuration and verify saved validation outcomes")
    main(overwrite=parser.parse_args().overwrite)
