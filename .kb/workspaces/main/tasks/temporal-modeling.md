---
id: 01a12838-40be-7fe1-928f-1c9ec15ecbfa
slug: tasks/temporal-modeling
title: "Temporal/sequence modeling (advanced tier)"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
blocked_by: [tasks/model-maturity]
---

User explicitly approved pursuing this too 2026-10-10, overriding the original suggestion to hold off for a separate go-ahead. Bigger architectural lift than tasks/model-maturity -- start after that lands since open-set detection and robustness eval are higher value per unit effort and this benefits from their groundwork (e.g. noisy scenario variants are a good stress test for a sequence model too).

Current model classifies each flow independently (i.i.d. MLP on single-flow features) -- it has no access to the sequence/timing structure within a session even though that structure is exactly what distinguishes e.g. beaconing (regular intervals) from a burst (tight clustering) from benign (sparse/random). Scope: a small sequence-aware model (e.g. GRU or simple windowed-aggregate features) operating per-session rather than per-flow, compared against the existing per-flow MLP the same way the MLP is compared against the rule baseline -- another rung on the same ladder, not a replacement. Vision's own gating language ('introduce temporal models...only when a documented data/task need and evaluation justify them') is satisfied by tasks/model-maturity's robustness work: if noisy/stealthy variants expose cases the per-flow model misses that session timing structure would catch, that's the documented need.

Keep the existing per-flow MLP (binary + multiclass) as the primary/default artifacts referenced by report.json; a sequence model is an additional comparison tier, not a replacement, consistent with the project's 'small baseline, compare complexity' philosophy.