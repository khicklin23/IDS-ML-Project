---
id: 01a12788-0dba-7063-8bcb-c532f509850b
slug: tasks/alert-store
title: "Alert generation and investigation storage"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
blocked_by: [tasks/inference-replay]
---

Future task, blocked by tasks/inference-replay. Read docs/PRODUCT_VISION.md (Alert investigation workspace). Turn scores+thresholds into severity-tagged alerts with history; support filtering/sorting, related flow/host timelines, supporting features, grouping into incidents, acknowledge/annotate, export. Separate storage/logic module from UI per vision's module-separation requirement. Agree file scope before editing. No implementation authorized yet.

### Implemented, 2026-10-10 (agent-2-env-setup)

Files: `src/ids_ml/alerts.py`, `tests/test_alerts.py`.

- SQLite-backed (stdlib `sqlite3`, no new dependency) `alerts` table: one row per flagged session_id with start/end time, flow_count, max/mean model_score, severity bucket (critical/high/medium/low from max_score), `scenario_ground_truth` (explicitly the generator's known-correct label, for evaluation/UI display only -- not something the current binary model "detected"), status, notes, incident_id.
- `generate_alerts_from_scores(scored_df, run_id, threshold=0.5)` consumes `inference.score_dataset` output directly -- alerts come from the model's `model_pred`, never from `baseline.rule_pred`, per the user's model-first priority (recorded in `context/roadmap` and local memory).
- `list_alerts` (filter by severity/status/run_id + sortable), `update_alert` (acknowledge/annotate, raises `KeyError` on unknown id), `group_into_incidents` (host + time-window grouping, documented as a simple heuristic, not real attack-chain correlation), `export_alerts` (csv/json).
- **Verification**: 5 new unit tests (severity bucketing, insert/filter round-trip, ack/annotate, incident grouping correctness, csv/json export) -- 39/39 project tests passing. Live smoke run chained generator → inference.score_dataset → alerts on the existing `data/generated/smoke` artifact: 4 alerts (3 critical, 1 high), grouped into 4 incidents, exported to `data/generated/smoke/alerts.csv`.

Not done (deferred, Stage 3+): real flow/host timeline drill-down views (that's UI, not storage), supporting-feature explanations beyond the aggregate stats above, alert volume at larger scale.

**Stage: First complete experience.** Together with tasks/inference-replay, this unblocks tasks/streamlit-ui.