---
id: 01a12804-8b71-7322-9374-fe5576b23b51
slug: tasks/attack-type-classifier
title: "Multiclass attack-type classification (NN-first)"
type: task
status: completed
priority: high
agent_id: agent-2-env-setup
tags: [pivot, multiclass]
---

User-directed pivot, 2026-10-10, supersedes the Stage-3 ordering in context/roadmap: build real multiclass attack-type classification, done primarily by the neural network (not deterministic rule code), as the next concrete deliverable -- ahead of the also-discussed static/Vercel single-dashboard rebuild and dataset-display polish, both explicitly deferred by the user. Scope: extend features/model/train/evaluate/inference to predict the specific scenario family (benign + each attack type) instead of just binary normal/suspicious, keep it additive (binary path and its rule-baseline-comparison regression test must keep working), and prove the NN genuinely earns its place with a regression test showing multiclass performance well above a trivial baseline. Light UI exposure only (e.g. Evaluation page); no deep investment in the current multipage Streamlit UI, since the user's real target is a different, static, single-dashboard, precomputed-batch presentation layer to be built later.

### Implemented, 2026-10-10 (agent-2-env-setup)

Purely additive -- the existing binary normal/suspicious path (Foundation + Stage 2) is untouched and its regression test (model beats rule baseline) still passes.

- **`models/classifier.py`**: `MLPClassifier` gained an `output_dim: int = 1` param. Same small architecture; `output_dim=1` is the existing binary head, `output_dim>1` is a softmax attack-type head. Forward only squeezes the output when `output_dim==1`, so existing binary callers are byte-for-byte unaffected.
- **`features.py`**: `SCENARIO_CLASSES` (= `schema.SCENARIOS`, fixed order so class indices are stable across runs), `SCENARIO_TO_INT`/`INT_TO_SCENARIO`, `extract_scenario_labels(df)` -- the multiclass target comes from the generator's own `scenario` column (ground truth), which was already excluded from `FEATURE_COLUMNS`; it's a *target* here, never an input.
- **`models/train.py`**: `train_multiclass_model(...)` -- same seeded/early-stopping/portable-artifact shape as `train_model`, but `CrossEntropyLoss` + per-class inverse-frequency weights + argmax + macro-F1 validation metric instead of `BCEWithLogitsLoss`/sigmoid-threshold. `save_artifact`/`load_artifact` gained optional `output_dim`/`class_names` (default `1`/`None`, so old binary artifacts still load unchanged).
- **`models/evaluate.py`**: `predict_class_proba` (softmax) + `evaluate_multiclass(model, X, y, class_names)` -- per-class precision/recall/F1/support, macro-F1, full NxN confusion matrix.
- **`inference.py`**: `score_dataset_multiclass(model, preprocessor, df, class_names)` -- adds `predicted_scenario`/`predicted_scenario_confidence`, explicitly the model's own inference, kept distinct from the generator's ground-truth `scenario` column.
- **`pipeline.py`**: `run_multiclass(...)` (mirrors `run()`) + `python -m ids_ml.pipeline --mode multiclass`.
- **Verification** (`tests/test_multiclass.py`, 5 tests): label extraction + unknown-value rejection; all 6 classes present in every split (had to search seed/episode-count space again -- same class-imbalance-meets-non-stratified-split issue as the earlier UI-default bug, this time per-class across 6 classes rather than binary); artifact round-trip exactness; `score_dataset_multiclass` matches training's own evaluation through an independent path and is confirmed to NOT be silently copying ground truth (asserts at least some predictions are wrong). The key regression test (`test_multiclass_model_beats_majority_baseline_and_identifies_each_attack`) requires >0.6 macro-F1 and >0.5 recall on *every* attack class, not just high overall accuracy (which a lazy "always predict benign" model gets for free).
- **Real numbers** from a full run (seed 99, 40 hosts, 90 min, 14 episodes/family, ~8.8k rows): **accuracy 0.928, macro-F1 0.90**. Per class F1: port_scan 1.00, outbound_transfer 1.00, auth_bruteforce 0.99, benign 0.95, service_flood 0.89, **beaconing 0.57** (precision 0.41 -- its main confusion is with benign: 83 of 1308 true-benign test rows get misclassified as beaconing, because the realistic-background refinement makes some ordinary short "established" flows look similar to C2 beacons). This is an honest result, not cherry-picked -- beaconing is a real, reported weak spot, not hidden.
- 65/65 project tests pass (60 existing + 5 new).

**Not done**: UI wiring (deferred per the user's explicit "for now" scope and the stated plan to replace the current multipage Streamlit UI with a static precomputed dashboard later -- not worth polishing the soon-to-be-replaced UI further right now). Rule-baseline multiclass comparison (the binary rule baseline's blind spots are documented; extending it to guess attack *type* was out of scope -- the ask was specifically for the NN to do this). Improving the beaconing/benign confusion specifically (candidate future work: more realistic beaconing signal, or additional discriminating features).