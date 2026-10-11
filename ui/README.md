# IDS Monitor

A static analyst workspace: light interface, alert/incident tables, inline investigations, model evaluation, a separate full raw-data browser, and an interactive model explanation page. No runtime backend or production dependencies.

## Preview

```bash
cd ui
npm run dev
```

Open http://127.0.0.1:4173. The development server serves files only. The deployed site requires no Python/ML service. Browser modules require HTTP rather than file://.

## Review workflow

Navigation opens three separate pages: Overview, Raw data, and Model explained. Alerts, incidents, and evaluation stay together on Overview. The explanation page includes layer exploration, saved training settings, scenario counts, and an adjustable teaching neuron; illustrative visuals are distinguished from measured results.

- Normal traffic is separated from attack families. The binary-results summary shows benign flows correctly left unflagged, detections, false positives and misses; these are held-out flow counts, not alert/session counts. Alert category filters distinguish attack-related records and benign false alarms using ground truth.
- Overview prioritizes alerts and incidents. Open a row to investigate inline beside the table; no modal dialog. Incidents use recorded backend group IDs, not invented attack-chain attribution.
- Alert details separate binary detection scores, multiclass guesses and generated ground truth. A benign ground-truth alert is identified as a false-positive example. A session link opens its full raw flows.
- Raw data opens raw.html: all 21,079 bundled flow records, 100 per page by default. Search covers every original field; family/outcome filters, sortable columns, page jump, and packet/provenance column controls are available. Each record includes the original ports and counters. No download is needed.
- Model evaluation is retained. Training histories and the decorative neural hero are removed.
- JSON import/export remains browser-local. Imported reports do not supply full raw datasets: links explicitly refer to the bundled dataset, and imported alerts do not get raw-session links. The raw page identifies its run and refuses mismatched report/raw exports.

The raw dataset has ground-truth labels, not per-flow model predictions. Embedded report flows are only a hundred-row sample; report alerts may also be sampled. Incident counts cover only embedded alerts. Static state is read-only.

## Refresh data

The bundled report and raw files are presentation copies of existing backend artifacts. No backend schema or implementation is changed. From ui/:

```bash
npm run sync-report
npm run sync-raw
npm run check
```

To use another report, pass its path to sync-report. The raw exporter uses the artifact_paths.dataset reference in the report by default (relative to the repository root), or accepts an explicit dataset directory:

```bash
npm run sync-report -- /absolute/path/report.json
../.venv/bin/python scripts/build-raw.py /absolute/path/dataset
```

The exporter requires the project's existing pandas/Parquet environment. It checks the dataset manifest, run ID and row count against the report, then writes static row-array chunks and a column manifest to raw-data/. The browser checks run/count/dataset generation time before loading them. Refresh both together. Only listed chunks are loaded; outdated unused chunks from a smaller refreshed run can be removed during packaging.

## Vercel

Use Root Directory ui, framework Other, no build command, and output directory dot (`.`). Ship report.json and the entire raw-data directory with the HTML/CSS/JS. No server functions or environment variables are required. No deployment was performed.

## Validation

`npm run check` runs JavaScript syntax checks, contract/filter checks, inline workspace event/render smoke checks, raw-data export checks, and raw-page navigation checks with Node's built-in runner. The lightweight DOM harness does not verify visual layout. Actual browser review is still needed for desktop/mobile appearance and keyboard focus behavior.

## Version comparison

Overview defaults to v3 and lets readers inspect v1/v2/v3. `versions.json` is a static copy of `data/generated/model_versions/comparison.json`. It compares attack-family guesses on the same test sessions. The version selector does not change the separate earlier report, its alerts, raw-data links, or Model explained page. v3 has no flow-level alert export in the comparison contract.

To refresh this comparison only, copy the backend comparison to `ui/versions.json` from the project root, then run the UI checks. Keep the full JSON shape; the page checks the schema, test counts, and class matrix totals. The scores are balanced class scores, not attack detection rates.
