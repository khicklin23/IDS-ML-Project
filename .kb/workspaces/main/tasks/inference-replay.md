---
id: 01a12788-045b-7d73-a8b4-f00a2ff31ad2
slug: tasks/inference-replay
title: "Inference, replay, and rule baseline"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
blocked_by: [tasks/train-evaluate]
---

Future task, blocked by tasks/train-evaluate. Read docs/PRODUCT_VISION.md (Detection and replay workspace). Load a saved model+preprocessing artifact; score generated telemetry; replay engine with play/pause/speed and a clock separate from wall-clock time; threshold tuning; a small rule-based baseline for comparison against the neural detector. Stays a separate module from UI and from alert storage. Agree file scope before editing. No implementation authorized yet.

### Implemented, 2026-10-10 (agent-2-env-setup)

Files: `src/ids_ml/inference.py`, `src/ids_ml/baseline.py`, `src/ids_ml/replay.py`, `tests/test_inference.py`, `tests/test_baseline.py`, `tests/test_replay.py`. Shared `evaluate_predictions()` factored out of `models/evaluate.py` so model and rule predictions are scored on an identical yardstick.

- **inference.py**: `load_model_for_inference` (named wrapper over `train.load_artifact`) + `score_dataset(model, preprocessor, df)` → copy of df with `model_score`/`model_pred`. Does not mutate input; verified it reproduces training's own held-out metrics exactly through an independent code path (`test_score_dataset_matches_training_evaluation`).
- **baseline.py**: deterministic rule (`outcome` rejected/timeout + short duration + few packets) — explicitly documented as comparison-only, never the primary detector, per user direction captured in `context/roadmap` (2026-10-10 priority note) and in local memory. Has a known, intended blind spot: it cannot see beaconing (which looks like ordinary short "established" traffic to a simple rule).
- **replay.py**: `ReplayController` with its own `sim_time` separate from wall-clock, `play`/`pause`/`set_speed`/`seek`/`advance(wall_dt)`/`events_up_to`/`window`. Auto-pauses at the end of the timeline.
- **Verification**: `test_model_outperforms_rule_baseline_overall` directly checks the user's stated priority — on the same held-out test set, the model's F1 ≥ rule F1 and model recall > rule recall (because the model catches beaconing and the rule cannot). This is now a regression test: if a future change made the rule baseline the de facto detector or let the model regress below it, this test would fail.
- 34/34 project tests passing after this task (up from 21).

**Stage: First complete experience.** Unblocks tasks/alert-store and (combined with it) tasks/streamlit-ui.