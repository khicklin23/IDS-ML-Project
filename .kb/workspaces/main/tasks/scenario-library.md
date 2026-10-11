---
id: 01a127bb-176f-7953-922f-5852d92c99b6
slug: tasks/scenario-library
title: "Expand scenario library and data realism"
type: task
status: completed
priority: medium
agent_id: agent-1-scenarios
blocked_by: [tasks/synthetic-data]
---

Stage 3 (Expanded simulation) work, pulled forward to run in parallel with tasks/streamlit-ui. Read docs/PRODUCT_VISION.md and context/roadmap (user priority note, 2026-10-10: push synthetic data toward realistic IDS telemetry fidelity; this is a step toward that, and toward eventual model-driven multiclass attack-type identification -- more scenario diversity now sets that up later). Add new scenario families to src/ids_ml/data/scenarios.py (candidates from the vision: service flood, unusual outbound transfer / data-exfil-like, lateral-movement-like; pick a few well-tested ones rather than all at once) and register them in src/ids_ml/data/schema.py (SCENARIOS tuple, SCENARIO_LABEL mapping -- all new families still map to label=suspicious; do not change label semantics). Add corresponding GenerationConfig fields in generator.py with safe defaults (e.g. 0 episodes) so existing calls/tests are unaffected unless explicitly opted into. Also push general feature realism: more plausible packet/byte-size and timing/inter-arrival distributions per service, closer to real flow-level IDS telemetry shape (CICIDS/NSL-KDD-ish), without inventing new schema columns. Extend tests/test_generator.py and tests/test_schema.py accordingly; keep determinism and the session-grouped no-leakage split passing. Acceptance: all existing tests still pass unmodified in behavior (only additive), new scenarios are deterministic and schema-valid, at least one new regression test per new scenario family.

### Coordination boundary (running in parallel with tasks/streamlit-ui, 2026-10-10)

Two agents working concurrently. File scope for THIS task: `src/ids_ml/data/scenarios.py`, `src/ids_ml/data/schema.py` (additive only -- extend `SCENARIOS`/`SCENARIO_LABEL`, don't rename/remove existing entries or change `FEATURE_COLUMNS`/`TARGET_COLUMN`), `src/ids_ml/data/generator.py` (additive `GenerationConfig` fields with defaults that preserve current behavior), `tests/test_generator.py`, `tests/test_schema.py`.

**Do not touch:** anything under `src/ids_ml/app/` (the other agent's UI work), `inference.py`, `baseline.py`, `replay.py`, `alerts.py`, `features.py`, `models/`, `pipeline.py`. If you think one of those needs a change to support a new scenario, stop and flag it in this task doc instead of editing it -- the other agent's UI is being built against their current interfaces live.

**Shared files:** avoid editing `docs/HANDOFF.md` / `context/progress.md` while the other agent is actively mid-session (both are single plain files, no lock). Keep your running notes in this task doc instead; add one short, additive entry to `docs/HANDOFF.md` only at the end of your session. Do not edit `tasks/streamlit-ui` or `context/roadmap`'s existing content -- append-only if you must touch `context/roadmap` at all.

Claim this with `git-kb set tasks/scenario-library status=active agent_id=<your-id>` (per AGENTS.md, `assign` will fail on an uncommitted doc -- that's expected, use `set`).

### Agent-1 implementation plan
Owner: agent-1-scenarios. Add opt-in service_flood and outbound_transfer families (lateral movement deferred), zero episode defaults, and realistic_background=False for service-aware background counters/durations. Preserve legacy output exactly when opt-ins are off. File scope: three data modules and two existing test files. Existing all-family test assumes SCENARIOS equals default-enabled families; extending registry with disabled families requires updating that assertion to preserve its original default-coverage meaning and adding explicit all-registry coverage. No detector/UI interfaces or binary label semantics change.

### Implementation checkpoint
Added service_flood (jittered short burst, one external→server pair, congested connection outcomes) and outbound_transfer (sequential upload-heavy HTTPS chunks, one internal→external pair). New peers are distinct even with missing role pools. Added realistic_background opt-in with service-specific counts, lognormal size variation, count-consistent IP-byte bounds and timeout behavior. All new/refined flows finish within their simulation window. Legacy scenario functions are untouched. Additive config validation rejects invalid new counts/flag. Baseline before edits: 39 passed in 73.32s. New families remain generated behavior, not evidence of real attack intent; defaults keep old output exactly.

### Verification and integration notes
Focused tests before config-validation additions: 32 passed in 10.77s. Ten-seed mixed generation smoke check passed schema, refined/new packet-byte constraints, in-window endpoints and pairwise session separation. Full suite running after final additions.

UI integration (read-only handoff): opt in via GenerationConfig(service_flood_episodes=2, outbound_transfer_episodes=2, realistic_background=True, seed=...). All fields are additive; new counts default to 0, realism defaults to False. The manifest includes these fields automatically. Scenario IDs are service_flood and outbound_transfer; both label suspicious. Schema version, feature columns, return types and binary target remain unchanged. No guarantee existing saved models detect new families; retrain and evaluate on expanded data before claiming coverage.

Deferred: lateral-movement sequences, multiclass model work, actual-traffic calibration. Legacy attacks retain their existing byte/timing quirks to avoid changing the UI demo and original model regression; new/refined rows have tighter constraints. Session-disjoint splitting is not host-disjoint across attack episodes (existing behavior); correcting that requires separately coordinated evaluation work. Code indexing deferred while UI agent is editing and repository metadata remains unavailable; no concurrent index rebuild attempted. No commits/pushes.

### Completed — agent-1-scenarios, 2026-10-10
Final verification: `.venv/bin/python -m pytest -q` → **60 passed in 77.34s** (original 39 plus 21 new cases). Ten-seed smoke check also passed. Only three scoped data modules and their two test files changed, plus this task and final additive shared notes. Defaults preserve legacy rows; optional realism/new families require explicit configuration. Ready for the UI agent to consume; no UI changes required.
