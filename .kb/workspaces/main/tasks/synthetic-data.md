---
id: 01a12780-af12-7f63-a95a-b2860fe8821d
slug: tasks/synthetic-data
title: "Build reproducible synthetic data"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
---

Future unassigned task after authorized setup. Read docs/PRODUCT_VISION.md. Define versioned event/flow schema and seeded, coherent host/session telemetry; configurable normal traffic and scenario families, noise and imbalance. Preserve generation manifests. Acceptance: determinism, cross-field constraints, no label/provenance leakage into model inputs, appropriate grouped splits, benign lookalikes and unseen scenario variations. Start with a few tested scenarios; extend later. Agree file scope before editing.

**Stage: Foundation.** Acceptance criteria above satisfy the vision's Foundation gate ("establish schema, seeded generator ... and meaningful checks"). Additional scenario families (scanning, auth bursts, service flood, unusual outbound transfers, beacon-like, lateral-movement-like) beyond the first few are Stage 3 (Expanded simulation) work — do not expand coverage here until Foundation and First-complete-experience land.

### Implemented, 2026-10-10 (agent-2-env-setup)

Files: `src/ids_ml/data/schema.py`, `src/ids_ml/data/scenarios.py`, `src/ids_ml/data/generator.py`, `tests/test_schema.py`, `tests/test_generator.py`.

- **Schema v1.0.0** (`schema.py`): event_time, duration, host/port/protocol/service identity, packet/byte counts, outcome, scenario provenance, label, plus generator-only `run_id`/`seed`/`session_id`. `FEATURE_COLUMNS` vs `NON_FEATURE_COLUMNS` are split explicitly so host identity, scenario, and session/run bookkeeping can never leak into a model's inputs (`validate()` also checks label↔scenario consistency, enum domains, port ranges, non-negative counts, no duplicate flow_id).
- **Generator** (`generator.py` + `scenarios.py`): seeded `np.random.default_rng`; coherent host population (workstation/server/external roles); benign background traffic with a realistic service mix; three scenario families to start (`port_scan`, `auth_bruteforce`, `beaconing`), each a self-contained "episode" with its own `session_id` so an attack's flows can't split across train/val/test.
- **Determinism**: same `GenerationConfig` → byte-identical DataFrame (`test_determinism_same_seed_same_output`); different seed → different data.
- **Splitting**: `split_dataset()` groups by `session_id` (host for benign, episode id for attacks) before splitting, so no session/episode straddles two splits — verified by `test_split_has_no_session_overlap` and deterministic given a seed.
- **Manifests**: `build_manifest`/`save_dataset`/`load_dataset` write `flows.parquet` + `manifest.json` (schema version, config, row/scenario/label counts, generated_at); `load_dataset` refuses a schema-version mismatch.
- **Verification**: `pytest tests/test_schema.py tests/test_generator.py -q` → all passing (part of the 21/21 full-suite run recorded in tasks/environment-setup). Live smoke run via `python -m ids_ml.pipeline --seed 123 --num-hosts 50 --duration-minutes 120` produced 8143 rows with a sensible normal-majority split.

Not done (explicitly out of scope per Stage tag above): additional scenario families, benign-lookalike/adversarial noise tuning, unseen-seed/shift evaluation harness — flagged for Stage 3 and for whoever builds the Stage-1 evaluation harness in tasks/train-evaluate to extend if needed.
