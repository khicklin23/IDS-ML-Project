---
id: 01a12780-b14c-7092-8178-48596a9ff6c1
slug: tasks/train-evaluate
title: "Train and evaluate classifier"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
---

Future unassigned task after synthetic data and authorized implementation. Read docs/PRODUCT_VISION.md. Start with a small reproducible PyTorch baseline, then configurable training, validation-based checkpointing, imbalance handling, experiment records, threshold selection and compatible model/preprocessing artifacts. Evaluate held-out and shifted scenarios, per-class metrics and false positives. Compare simpler baseline before adding complexity. Training logic stays separate from UI. Agree file scope before editing.

**Stage: Foundation** for the small baseline + held-out evaluation + meaningful checks. Configurable architectures, checkpoint selection, and experiment-comparison records belong to Stage 3 (Expanded simulation) and should wait for tasks/inference-replay and the first vertical slice to exist, so there is a real UI/workflow to compare runs against. Temporal/anomaly modeling is explicitly Stage 4 (Advanced research) — do not start it without documented Foundation/Stage-2 results motivating it.

### Implemented, 2026-10-10 (agent-2-env-setup)

Files: `src/ids_ml/features.py`, `src/ids_ml/models/classifier.py`, `src/ids_ml/models/train.py`, `src/ids_ml/models/evaluate.py`, `src/ids_ml/pipeline.py`, `tests/test_train_evaluate.py`.

- **Feature prep** (`features.py`), kept as its own module per the vision's component-separation rule, not inside generator or train: `ColumnTransformer` (StandardScaler on numeric, OneHotEncoder on protocol/service/outcome) fit on `FEATURE_COLUMNS` only — host identity, scenario, session/run bookkeeping never reach it. Fit on train split only; val/test only ever call `.transform`.
- **Model** (`classifier.py`): small MLP (Linear→ReLU→Dropout→Linear→ReLU→Linear, default hidden_dim 32) — deliberately small baseline to compare future complexity against, per vision.
- **Training** (`train.py`): seeded (`set_seed`) for numpy/random/torch; validation-based early stopping (tracks best val loss, restores best `state_dict`, configurable patience); optional `pos_weight` class-weighting for imbalance; portable artifacts via `save_artifact`/`load_artifact` (model state dict + fitted preprocessor + metadata.json with schema_version, feature lists, train config, metrics — `load_artifact` refuses a schema-version mismatch, same pattern as the dataset loader).
- **Evaluation** (`evaluate.py`): accuracy, precision, recall, F1, confusion matrix, false-positive rate on held-out test split.
- **Pipeline** (`pipeline.py`): wires generate→split→preprocess→train→evaluate→save into one callable/CLI (`python -m ids_ml.pipeline`) without hiding any training logic inside UI callbacks.
- **Verification**: `pytest tests/test_train_evaluate.py -q` passing (part of 21/21 full suite, see tasks/environment-setup); checks held-out metrics are in [0,1] with no NaNs, recall clears a sanity floor, and a saved-then-reloaded artifact produces byte-identical evaluation metrics (round-trip correctness, not just "it runs"). Live run: `python -m ids_ml.pipeline --seed 123 --num-hosts 50 --duration-minutes 120` → held-out test accuracy 0.998, precision 0.991, recall 1.0, FPR 0.002 on 1185 held-out rows. That number reflects separation between these three synthetic scenario signatures and normal traffic, not real-world intrusion detection accuracy.

Not done (explicitly out of scope per Stage tag above): configurable architecture search, checkpoint/run comparison records, threshold-tuning UI, robustness to distribution shift/unseen scenarios — Stage 3 work once tasks/inference-replay and the first vertical slice exist to make comparisons meaningful.
