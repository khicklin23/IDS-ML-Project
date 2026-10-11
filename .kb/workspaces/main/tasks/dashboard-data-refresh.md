---
id: 01a12b97-b649-7702-9181-d03effc96a9b
slug: tasks/dashboard-data-refresh
title: "Refresh stale Overview and raw dataset"
type: task
status: completed
priority: medium
agent_id: agent-1-refresh
---

Refresh UI to latest complete showcase_50k report and matching raw export. Support additive report 1.1, distinguish v4 comparison from binary alert bundle, update default sync source and snapshot-sensitive checks. No retraining or fabricated v4 alerts.
## Completed and remaining backend integration

Confirmed stale UI report 1.0 from 2026-10-10, 21,079 flows. Latest complete report is showcase_50k, 2026-10-11, 49,014 flows, report1.1/schema1.2. Synced exact report plus 50 matching raw chunks. UI now supports additive report1.1 and default sync source showcase_50k. Displayed report/schema metadata and explicit flow-detector vs v4 comparison label. Updated model dataset wording and UI docs. Snapshot-sensitive tests derive counts/pages from report and raw manifest, preserving behavior assertions. npm --prefix ui run check: six files pass, report equality confirmed, raw uniqueness/provenance tested.

v4 overview integration is not available in backend artifacts: versions.py saves comparison and model artifacts but no dataset Parquet, version-specific alerts or binary v4 detector/report. Did not mix session family scores into binary metrics or manufacture alerts. Future backend task must export a defined latest-version scoring/report/raw contract before Overview can claim v4. No backend edits, retraining, commits or deployment.

## v4 integration completed after backend export

Read model-versions export handoff and real showcase_v4/report.json. Switched default sync to showcase_v4; copied exact report, refreshed corrected comparison, exported 87,505 raw rows into 88 static chunks. Adjusted exporter run validation for two source seeds (run-7/run-1007) under session-7 report, and report-time fallback for combined manifest lacking generated_at. Raw page checks matching fallback time, tests preserve source run IDs and unique flow IDs. Alert links keep namespaced sessions.

Overview now explains one v4 session detector, no independent binary head, score=1-P(benign), non-benign family triggers alerts, inherited flow metrics. Model explained now displays GRU memory architecture from detector metadata, one shared six-output model, no fabricated cutoff for null threshold, current v4 training explanations. Retained legacy report-render branches. Updated UI guide and source-sensitive checks. npm --prefix ui run check passes all six files; exact report equality confirmed. Backend and backend tests untouched. No browser visual QA, commits/pushes/deployment. Previous backend integration blocker resolved.

## Cached validator fix

User saw old exact error expected1.0 despite local validator accepting1.1. Added release query v4-20261011-2 to every page CSS/JS reference and every browser module import, so refreshed HTML loads new module URLs. Vercel static assets revalidate; model/comparison/raw reads bypass cached JSON. UI checks pass six files with query-based Node imports. UI guide explains release bumps and hard refresh. Local HTTP probe unavailable under sandbox socket restriction; no server/deployment claimed verified.
