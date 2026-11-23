# ECE 539 SMS spam prototype — progress phase

This repository contains the November 22 preliminary experiments for the ECE 539 G22 project progress report. A data pipeline, TF-IDF/logistic baseline and word-level CNN pilot are implemented. This is not a completed final system, full experimental comparison or final test evaluation. Team assignments remain to be confirmed. The checkpoint was developed with AI assistance.

## Official data and provenance

- Dataset: **SMS Spam Collection**, Almeida and Hidalgo (2011), UCI Machine Learning Repository. The repository donation date is 2012; the official dataset citation uses 2011.
- Metadata: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- DOI: https://doi.org/10.24432/C5CC84
- Official download: https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip
- License shown by UCI: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/ . Preserve source attribution when sharing derived work.
- Downloaded for this prototype on 2026-11-22. ZIP SHA-256: `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`.
- `artifacts/data_audit.json` records download URL, DOI, license, ZIP and individual extracted-file hashes, parsing rules, exclusions, split counts, overlap checks, and environment. `artifacts/split_manifest.json` records every split's stable row IDs and CSV SHA-256.

The original file is tab-delimited, with one message per physical LF-delimited line. The parser splits only at the first tab; quotation marks inside a message are literal. It does not interpret the source as a quoted CSV. The actual audit found **5,574 valid rows**, matching the UCI catalog: 4,827 ham and 747 spam.

## Cleaning and partitions

Text is lowercased and whitespace is collapsed. Messages with identical normalized text are grouped before splitting. A consistent-label group contributes one row; all rows of any conflicting-label group would be excluded. Empty or malformed records would also be excluded.

The audit removed **415 redundant rows in 290 consistent-label duplicate groups**, leaving **5,159 unique messages**: 4,517 ham and 642 spam. It found no empty, malformed, or conflicting-label records. No SMS message text is needed in report output.

| Partition | Ham | Spam | Total |
|---|---:|---:|---:|
| Train | 3,162 | 450 | 3,612 |
| Validation | 678 | 96 | 774 |
| Reserved test | 677 | 96 | 773 |

The target ratio is 70%/15%/15%, stratified by label with seed **539**. Within each class, rows are initially SHA-256 sorted, shuffled with `random.Random("539:" + label)`, and allocated by the largest-remainder method. Ties use train, validation, then test order. Exact counts differ slightly from percentage targets due to rounding. No normalized text or row ID occurs in more than one partition.

CSV contract: `data/processed/{train,validation,test}.csv`, UTF-8, columns `row_id,label,text,source_line_ids,duplicate_count`. `row_id` is SHA-256 of normalized UTF-8 text; labels are `ham`/`spam`; `source_line_ids` is a JSON array of one-based original line numbers. The shared `src/data_interface.py` loads checksum-verified train/validation rows and maps ham=0, spam=1. It rejects test loading.

**Test status: reserved, not evaluated.** Test access has been limited to creation, split bookkeeping, and file checksums. No test predictions, threshold selection, or numeric test performance metrics have been produced.

## Initial baseline and actual validation results

The baseline uses training-only word TF-IDF (unigrams/bigrams, at most 20,000 features, sublinear term frequency, L2 normalization) and logistic regression (`C=1`, balanced class weights, `liblinear`, seed 539). Its vocabulary and IDF are fitted only on training text. No hyperparameter search was performed.

The operating threshold is selected only on validation data by maximizing spam F1. Candidates cover every distinct validation prediction pattern; ties prefer lower ham false-positive rate and then a higher threshold. Selected threshold: **0.518305307757728**. Spam predictions use probability greater than or equal to the threshold.

| Validation metric | Fixed threshold 0.5 | Selected threshold |
|---|---:|---:|
| Spam precision | 91.75% | 94.68% |
| Spam recall | 92.71% | 92.71% |
| Spam F1 | 0.9223 | 0.9368 |
| Accuracy | 98.06% | 98.45% |
| Ham false-positive rate | 1.18% | 0.74% |

At the selected threshold the confusion counts are TN=673, FP=5, FN=7, TP=89, on 774 validation messages (96 spam). `artifacts/baseline_metrics.json` is the authoritative machine-readable result; `artifacts/baseline_validation_predictions.csv` contains only row IDs, labels, and validation probabilities/predictions. `models/tfidf_logistic.joblib` contains the trained pipeline and threshold.

These are **preliminary development results** from one split. The selected-threshold metrics use the same validation set that chose the threshold and therefore are optimistic estimates for development, not an unbiased test estimate. Historical SMS data may not represent current spam. No deployment or final generalization claim is supported yet.

## Real word-level CNN pilot

`train_cnn.py` trained one CPU model with seed 539 using only the same training and validation partitions. A training-only regex word vocabulary contains 3,108 IDs including padding and unknown tokens. Sequences are padded or truncated to 128 tokens. The model uses a trainable 64-dimensional embedding, 64 Conv1D filters of width 3, ReLU, max pooling over valid token-centered positions, dropout 0.3, and a linear output. Training uses unweighted BCEWithLogitsLoss and Adam (learning rate 0.001, batch size 64). Sigmoid converts logits to probabilities. This is a word-level CNN, not a character-level model.

The single run completed 12 epochs; the checkpoint with minimum validation loss was epoch 11. At fixed threshold 0.5, spam precision is **100.00%**, recall **82.29%**, F1 **0.9029**, and accuracy **97.80%**. Confusion counts are TN=678, FP=0, FN=17, TP=79. Ham false-positive rate is zero on these 678 validation ham messages; this does not imply a zero population error rate.

A separate validation-only threshold exploration chose 0.35 from a prespecified 0.05-step grid, with F1 **0.9071**. This tuned result is optimistic, and its search rule differs from the baseline's. Use fixed 0.5 for the initial comparison and standardize the threshold-selection protocol before the final tuned comparison. Even the fixed-0.5 CNN score uses a validation-selected checkpoint.

Exact configurations, checks, metrics, predictions and learning curves are in `artifacts/cnn_pilot/`; weights and vocabulary are in `models/cnn_pilot/`. The pilot has 211,329 parameters. Padding, no-token input and test-access invariants passed. Three-seed experiments, dropout ablations and final held-out evaluation have **not** been done.

## Notebook

Open `progress_notebook.ipynb` from this project directory to inspect the audited data, real saved metrics, confusion matrices and learning curve. Its saved outputs come from reading the actual experiment artifacts; the notebook does not pretend to have retrained them. Set `RUN_TRAINING = True` only when you want to rerun and replace those artifacts. This explicitly passes `--overwrite` to the CNN script. The notebook itself performs no GitHub upload or course submission.

## Reproduction

Python 3.11.11 was used. Install `requirements.txt` for the complete project, or `requirements-baseline.txt` for only the baseline. CPU PyTorch 2.6.0+cpu was used; `torch==2.6.0` in the requirements accepts that local build. Launch the notebook with a Jupyter client using this environment.

Run from this directory:

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py --download
python src/train_baseline.py
python src/verify_progress.py
python train_cnn.py --overwrite
```

The preparation command downloads only when the ZIP is absent, verifies the locked ZIP checksum, and recreates the deterministic split files. It refuses an unexpected checksum. Preparation never trains a model. The training command loads only train and validation CSVs. The verification command checks source/split hashes and ID overlap, confirms the test-access guard, confirms a training-only refit gives the saved TF-IDF vocabulary/IDF, and reproduces saved validation predictions and metrics. It writes `artifacts/verification.json`.

Use your Python 3.11 interpreter. Paths are relative to the project directory. The CNN refuses to replace existing results unless `--overwrite` is explicitly provided. The archived run was repeated once after adding this guard option, solely to check reproducibility; no model settings were changed or selected from repeated runs.

Next steps are to review validation errors, compare the word-level CNN across seeds and dropout settings under a documented protocol, freeze model/threshold choices, and perform one held-out evaluation per final run after development is complete. A user-facing demo and final report/slides remain unfinished. Proposed team assignments must be confirmed by the team.
