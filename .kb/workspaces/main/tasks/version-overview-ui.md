---
id: 01a12b6a-9aea-79a2-b45d-9dc99c6f4d69
slug: tasks/version-overview-ui
title: "Version progression on Overview"
type: task
status: active
priority: medium
agent_id: agent-1-version-ui
---

UI-only: read saved comparison, default to v3, compare v1/v2/v3 with plain language and session-level statistics; distinguish older alert snapshot. Preserve backend and data artifacts. Verify selection, validation, and existing UI checks.
## Completed

Added UI-only version comparison: ui/versions.json copies the saved backend comparison unchanged; versions-data.mjs validates it, versions.js renders v3 by default with selectable progression cards, session statistics and a shared comparison table. Plain language defines flow, session, GRU and macro F1. v2 strongest result and v3 limitations visible; no attack-catch claims from class scores. Earlier report metadata/controls and investigation snapshot separated below comparison to prevent attributing old alerts to v3. Backend/report/raw exports and model explanation page untouched. UI guide updated. npm run check: all six files pass; selection, defaults, arithmetic and invalid contract checks included. HTML duplicate IDs/assets checked. No browser visual QA, commits, pushes or deployment.
