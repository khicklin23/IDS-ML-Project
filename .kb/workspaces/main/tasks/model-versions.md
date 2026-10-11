---
id: 01a12b39-c00c-70b3-a495-ab16642e2e14
slug: tasks/model-versions
title: "Model version registry: v1 -> v2 -> v3 progression"
type: task
status: completed
priority: high
agent_id: agent-2-env-setup
---

User-directed, 2026-10-11: version the attack-type classifier explicitly (v1/v2/v3) rather than just overwriting it, so improvement over time is measurable on a shared eval harness, not just claimed. Three versions planned for this round:

v1: per-flow MLP baseline, trained on one normal (non-stealth) generation run. This is what tasks/attack-type-classifier and tasks/model-maturity already built and validated -- repackaged as the v1 artifact under the registry, not rebuilt from scratch.
v2: same per-flow MLP architecture, but the training recipe validated in tests/test_robustness.py as standard practice: mixed normal+stealth training data. Already proven in testing to recover most stealth-robustness lost by v1 without sacrificing normal-case performance. Repackaged as v2, not new modeling work.
v3: session-level sequence model (small GRU) operating on a session's ordered sequence of flows instead of one flow at a time. New work -- directly motivated by tasks/model-maturity's open-set finding that beaconing's signal (timing regularity across many flows) is structurally invisible to any per-flow feature set, however augmented. Compared against v1/v2 on the same axis: roll per-flow models' predictions up to session-level (majority vote) so all three are judged on session-level accuracy/macro-F1, not an apples-to-oranges comparison.

Deliverable: a shared comparison harness/report showing all three versions' metrics side by side (session-level accuracy, macro-F1, per-class, and stealth-robustness where applicable) so the progression is visible as data, not prose. Keep each version's artifact on disk (don't overwrite) so the comparison is reproducible later. No UI work; report.json contract stays as-is unless a real need to extend it emerges (flag in tasks/static-dashboard-ui if so).

### Built, 2026-10-11 (agent-2-env-setup)

Files: `src/ids_ml/models/sessions.py` (`build_session_sequences` -- turns a flow DataFrame into padded per-session sequences using the *same* preprocessed features v1/v2 use, so only architecture differs), `src/ids_ml/models/sequence.py` (`SequenceClassifier`: one `nn.GRU` + linear head over packed variable-length sequences; `train_sequence_model` mirrors `train_multiclass_model`'s shape -- seeded, validation-based early stopping, history log), `src/ids_ml/versions.py` (`run_comparison`: generates one normal + one stealth dataset, trains all three versions, rolls v1/v2's per-flow predictions up to session-level via majority vote for a fair comparison against v3's native session-level output, saves every artifact + `comparison.json`). Refactored `models/evaluate.py` to expose `multiclass_metrics_from_predictions`/`multiclass_metrics_from_probs` as shared scoring so the per-flow model, the session rollup, and the sequence model are all scored identically.

**Scale note**: session-level sample size is sessions, not flows -- the configs used everywhere else in this project (tens of episodes) give only ~1 session per class at the test split, useless for evaluation. Tuned to `num_hosts=150, episodes_per_family=50` (duration 180min): ~360 total sessions, ~34 train / 8 test per attack class, ~78 train / 16 test benign -- small by ML standards but enough for a real, if noisy, comparison. `tests/test_generator.py`'s existing per-row-count configs are unaffected; this is purely a `versions.py`-level tuning choice.

**Result**, confirmed across two runs (`tests/test_model_versions.py`'s reduced-epoch config, and a full-default-epoch run saved at `data/generated/model_versions/comparison.json`): all three handle normal (non-stealth) traffic perfectly at this scale (session macro-F1 1.0 for all three -- 56 normal test sessions). On stealth-evasion traffic neither v1 nor v3 ever saw in training:

| | v1 (baseline) | v2 (mixed training) | v3 (session GRU) |
|---|---|---|---|
| reduced-epoch run | 0.55-0.57 | 1.0 | 0.64-0.71 |
| full-epoch run | 0.598 | 1.0 | 0.619 |

**v2's fix is large and consistent**: ~1.0 both times -- the training-recipe fix from tasks/model-maturity, now confirmed at session level too. **v3's fix is real but smaller and noisier**: it beat v1 in every run (never worse), but the margin ranged +0.02 to +0.16 across runs/epoch budgets rather than a single clean number. Being honest about why: the stealth test split is only 56 sessions (~9 per attack class) -- small enough that the exact margin is sensitive to training duration and random seed, even though the *direction* (v3 > v1) held every time it was checked. Reported as a real-but-modest, somewhat noisy effect, not a single cherry-picked number. The underlying mechanism (seeing a session's whole sequence gives some inherent robustness to pacing changes, for free, without ever training on stealth data) is the actual finding -- the precise percentage is secondary and would benefit from a larger stealth-test sample if this becomes a headline number anywhere.

Two different, real improvement axes: v2 improves via better training data, v3 improves via better architecture -- both beat v1, independently, neither a replacement for the other.

**Natural v4 (not built, flagged for later if wanted)**: v3's architecture + v2's mixed-stealth training recipe together -- the two axes aren't mutually exclusive, might compound. Noted, not pursued this round per the "probably 3 variations" scope.

85/85 tests passing. Artifacts: `v1/`, `v2/` (full `save_artifact` bundles: model+preprocessor+metadata), `v3_model.pt`+`v3_metadata.json`, `comparison.json` -- all under one `out_dir`, nothing overwritten between versions. Status: **completed**.