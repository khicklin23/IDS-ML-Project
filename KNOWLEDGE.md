# Sona technical knowledge

Verified against the repository implementation and bundled artifacts on **2026-10-13**. This document is a technical reference for developers and readers who want implementation details. Values below describe a saved snapshot, not guarantees about future runs. If source code or artifacts change, inspect them again before relying on the numbers.

## 1. Essential facts

- **Sona** is the public name of internal model version `v4`. `v1`, `v2`, and `v3` display as Beta 1, Beta 2, and Beta 3. Keep internal keys stable.
- The current public detector is **one six-class, session-level GRU**, not two separately trained networks. Normal-versus-suspicious evaluation and family classification are two views of its output.
- Traffic is generated in Python. It resembles flow-level telemetry, but is not packet capture, payload inspection, real malware execution, or a production network sensor.
- The public dashboard is static HTML/CSS/JavaScript. It reads saved JSON. It does not run PyTorch, train, capture traffic, or perform live inference in the browser.
- The public checkpoint behind Overview and the retrained timing-study candidates behind Sona Progression are **different artifacts**. Better experimental results do not automatically replace the public model.
- Model inputs exclude labels, scenario names, addresses, host IDs, and run identifiers. Session IDs and event times organize sequences but are not direct feature columns.
- Session construction is supplied by the synthetic generator. Reliable real-world session formation is an additional engineering problem, not an implemented capability demonstrated by these results.

## 2. Repository architecture

| Module or directory | Responsibility |
| --- | --- |
| `src/ids_ml/data/schema.py` | Schema version, feature allowlist, class registry, dtypes, validation |
| `data/scenarios.py` | Hosts, normal activity, attack episodes, optional background refinements |
| `data/generator.py` | Seeded generation, manifests, Parquet storage, run combination, session splits |
| `features.py` | Numeric/categorical preprocessing, targets, persisted timing transforms |
| `models/classifier.py`, `models/train.py` | Earlier connection-level MLP models and artifact persistence |
| `models/sessions.py`, `models/sequence.py` | Ordered session tensors, GRU training and prediction |
| `models/evaluate.py` | Binary and multiclass metrics plus calibration helpers |
| `pipeline.py` | Earlier binary or multiclass training workflows |
| `versions.py` | Historical four-version architecture/training comparison |
| `report.py` | Earlier two-network `build_report` and current one-GRU `build_session_report` |
| `inference.py` | Earlier connection-level scoring helpers |
| `baseline.py` | Simple deterministic comparison rule |
| `alerts.py` | Model-output aggregation, SQLite storage, severity, incident grouping |
| `replay.py` | Simulated-time playback controller, separate from wall-clock time |
| `robustness.py`, `openset.py` | Distribution-shift and unknown-family experiments |
| `experiments/` | Timing benchmarks, joint six-class candidates, stress studies, artifact audits |
| `app/` | Separate Streamlit workbench code; not the deployed static dashboard |
| `ui/` | Public static UI, contracts, presentation data, exports, release builder, Node checks |
| `tests/` | Python generator/model/report/experiment regression checks |

Paths in the module column are relative to `src/ids_ml/` unless otherwise stated. Python packaging uses the `src` layout, setuptools, and Python >=3.10. `requirements.txt` lists NumPy, pandas, scikit-learn, PyTorch, Streamlit, Plotly, and pytest, with the PyTorch CPU extra index. Parquet operations also require an available pandas Parquet engine. Existing `.venv` is local and is not part of a clone.

## 3. Data contract and generation

Current dataset schema: **`1.2.0`**. `SCENARIOS` has a fixed order, also used for output indices:

| Index | Scenario | Binary label |
| ---: | --- | --- |
| 0 | `benign` | `normal` |
| 1 | `port_scan` | `suspicious` |
| 2 | `auth_bruteforce` | `suspicious` |
| 3 | `beaconing` | `suspicious` |
| 4 | `service_flood` | `suspicious` |
| 5 | `outbound_transfer` | `suspicious` |

The legitimate input columns are:

```text
duration_s, src_port, dst_port,
protocol, service,
packets_sent, packets_recv, bytes_sent, bytes_recv,
outcome, prior_gap_s
```

`protocol`, `service`, and `outcome` are categorical; the remaining eight are numeric. One-hot encoding means transformed feature count is not necessarily eleven. Derive it from the saved preprocessing or actual tensor shape.

Other row fields include schema version, run ID, seed, flow ID, event time, session ID, host IDs, source/destination IPs, scenario, and label. These are provenance, targets, grouping, or display fields. There are no MAC-address or packet-payload inputs in this schema.

`generate_dataset(GenerationConfig(...))` uses a seeded NumPy generator, constructs hosts and flows, sorts activity, calculates `prior_gap_s` within each session, coerces dtypes, validates, and returns a DataFrame and manifest. The first flow's prior gap is zero. Repeated generation with identical settings produces identical flow data; manifest timestamps can differ.

Default config is 40 hosts, 60 simulation minutes, background rate 60/minute, and three episodes each of scanning/authentication/beaconing. Service flood and outbound transfer default to zero; `realistic_background` and `stealth_mode` default to false. Therefore a default generator call is not equivalent to the larger six-family showcase.

Each attack episode is one single-category session. Normal background sessions are grouped by source host. This construction is convenient for supervised experiments but does not demonstrate discovery of true incident boundaries in real networks.

### Lower-rate variants

`stealth_mode=True` changes several distributions; it is not a playback-speed setting:

| Family | Standard | Lower-rate variant |
| --- | --- | --- |
| Scan | 30–199 ports, 5–60-second span | 10–39 ports, 300–1,800-second span |
| Authentication | 30–149 attempts, 30–300-second span | 5–19 attempts, 1,800–7,200-second span |
| Flood | 100–249 flows, 2–15-second span | 20–59 flows, 30–120-second span, more established outcomes |
| Outbound transfer | 60–600-second span, throughput distribution centered at 500 packets/s | 900–3,600-second span, centered at 120 packets/s |
| Beaconing | 45–120-second interval, jitter fraction 0.1 | 180–600-second interval, jitter fraction 0.35 |

Some spans are limited by simulation duration. Values are sampled, not fixed per episode. Lower-rate variants retain recognizable per-flow features, so a successful classifier does not necessarily require memory to recognize them. Upload-heavy telemetry alone cannot establish malicious intent.

### Identity and splitting

Independent generation runs reuse local session names. `combine_runs` prefixes each session with its source `run_id` before concatenation. Plain concatenation can silently merge unrelated episodes. Canonical identity checks must account for the run as well as the session.

`split_dataset` assigns whole sessions to train/validation/test, allocating separately by scenario. Defaults are approximately 70/15/15 percent of sessions, not exact percentages of flows. Very small family populations receive best-effort allocation. Current report generation splits the standard and lower-rate runs separately, then combines matching partitions using `combine_runs`.

Fit preprocessing on training data only. Never fit encoders or scalers on the full dataset before splitting. Exclude labels and generator provenance from features even if they improve a score.

## 4. Preprocessing and timing

`build_preprocessor` uses `StandardScaler` for numeric features and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features. `transform_features` returns float32 arrays. Feature ordering and category vocabularies belong to the fitted artifact; inference must reuse them.

Timing modes:

- `absolute`: use `prior_gap_s` in seconds before standardization. This is the fallback for the current public artifact, which predates explicit timing-mode metadata.
- `relative`: divide each gap by the mean positive gap of that complete session. This retains within-session shape while reducing sensitivity to overall pace.
- `hybrid`: retain absolute gaps and add a relative-gap feature.
- `hybrid_log`: retain `log1p` of absolute gaps plus the relative-gap feature.

Non-absolute modes persist a `SessionTimingPreprocessor` wrapper with the fitted encoder. Relative normalization uses complete closed sessions and does not read labels. Its session-wide scale is unavailable for a partial online session without a different design. Do not describe this as an implemented streaming transform.

## 5. Architectures and training

### Earlier connection models

`MLPClassifier` defaults to two hidden layers: 32 and 16 units, ReLU activations, and dropout 0.1 after the first hidden layer. Binary output width is one, trained with `BCEWithLogitsLoss`; multiclass output width is the class count, trained with cross-entropy. Earlier `build_report` trains independent binary and multiclass networks. Their answers may disagree.

Historical Beta 1/2 family comparisons classify individual connections and take a majority vote for a session. Voting combines predictions without using their order. Do not confuse this with a GRU's learned memory.

### Current session model

`SequenceClassifier` has one PyTorch GRU layer, default hidden dimension 32, followed by `Linear(hidden_dim, num_classes)`. Inputs have shape `[sessions, max_length, transformed_features]`. `build_session_sequences` sorts chronologically, pads with zeros, returns true lengths and targets, and caps sequences at the caller-supplied maximum.

`pack_padded_sequence` prevents padding from acting as genuine observations. The final hidden state feeds the output layer. Forward returns logits; prediction applies softmax. There is no second detector head, attention mechanism, or MLP-style dropout in this implemented GRU.

The helper's default maximum is only 50 flows. Public report generation explicitly chooses a data-sized maximum with headroom; the saved public maximum is **274**. Other callers must supply a suitable cap. Earlier truncation mistakes are why this distinction matters. The current sizing helper inspects session lengths across partitions; preprocessing parameters and weights are still trained without test labels. Length sizing is a known protocol detail, not proof that every future session fits.

For each training batch: forward pass -> class-weighted cross-entropy -> backpropagation -> Adam update. Sessions are shuffled between batches, but flows retain their within-session order. GRU hidden state is not persistent across unrelated sessions.

Sequence class weights are proportional to `count^(-class_weight_power)`, normalized to sum to the number of classes. Power 1 means inverse frequency; power 0.5 means softer square-root weighting. Counts are session counts for sequence training. Learned network weights and loss class weights are different concepts.

Checkpoint selection uses lowest validation loss, with patience-based early stopping and an optional minimum training period. Validation macro F1 is logged but does not select the checkpoint. Saved public settings: seed 7, hidden size 32, learning rate 0.001, maximum 60 epochs, patience 10, batch size 32, class weighting enabled. The older saved config omits newer fields such as `min_epochs`, `class_weight_power`, and `timing_mode`; do not claim current function defaults were necessarily used to produce older weights.

## 6. Current inference and alerts

The authoritative path is `report.build_session_report`:

```text
standard + lower-rate datasets
  -> whole-session partitions
  -> train-fitted preprocessing
  -> ordered session tensors
  -> one GRU -> six-class softmax
  -> winning category per session
  -> broadcast category, confidence, and suspiciousness to its flows
  -> evaluation, alert aggregation, SQLite/CSV/report exports
```

For class probabilities `p`:

```python
predicted_category = class_names[argmax(p)]
predicted_scenario_confidence = max(p)
model_score = 1 - p[class_names.index('benign')]
model_pred = int(predicted_category != 'benign')
```

The flag is based on the winning category, not `model_score >= 0.5`. These are not equivalent in a six-class model. For example, benign can have probability 0.4 and still beat every individual attack category, despite total attack probability being 0.6.

`binary_model` and `multiclass_model` remain JSON keys for compatibility, but both describe the same GRU in a session report. Inspect `detector.kind`, `detector.note`, and `artifact_paths.sequence_model` rather than assuming two models from those key names.

`generate_alerts_from_scores` finds flagged sessions from `model_pred`, then aggregates their rows. Its threshold argument does not override those flags. Alerts retain predicted family and generator ground truth in separate fields. One flag per flow is not one alert per flow.

Severity is a deterministic mapping from maximum suspiciousness score: >=0.95 critical, >=0.80 high, >=0.60 medium, otherwise low. This is score-based presentation, not independently validated business impact. Softmax confidence is not a calibrated guarantee.

Incidents group alerts from the same source host within a default 300-second gap from the current incident end. This is deterministic correlation, not learned attack-chain attribution. Static incident displays only have access to the embedded alert sample.

`baseline.rule_score` flags rejected/timed-out connections shorter than one second with at most three sent packets. It remains in reports/tests but was removed from the public performance cards. It is a deliberately narrow baseline, not a benchmark against enterprise IDS products.

## 7. Public snapshot and measurement units

Authoritative public presentation: `ui/report.json`, copied from `data/generated/showcase_v4/report.json`. Current report version **1.1.0**, schema **1.2.0**, run `session-7`. The combined snapshot uses source seeds 7 and 1007, 200 hosts configured per run, 180 simulation minutes per run, and 100 episodes per attack family per run. `realistic_background` is false in this snapshot; newer experiments can enable it.

| Quantity | Snapshot |
| --- | ---: |
| Total flows | 87,505 |
| Training flows | 61,280 |
| Validation flows | 13,296 |
| Test flows | 12,929 |
| Normal flows, complete dataset | 21,582 |
| Suspicious flows, complete dataset | 65,923 |
| Total generated test alerts | 151 |
| Embedded alert sample | 50 |

The binary test confusion counts are TP=9,879, TN=2,974, FP=76, FN=0. All 76 misclassified normal flows receive the beaconing category. They may belong to fewer sessions; do not describe these as 76 separate false alerts.

| Metric | Value |
| --- | ---: |
| Detection accuracy | 99.4122% |
| Precision | 99.2366% |
| Recall | 100% |
| Detection F1 | 99.6168% |
| False-positive rate | 2.4918% |
| Classification macro F1 | 99.5451% |

Definitions: precision=`TP/(TP+FP)`; recall=`TP/(TP+FN)`; F1 is their harmonic mean; false-positive rate=`FP/(FP+TN)`. Macro F1 averages per-category F1 equally. Accuracy counts correct predictions divided by total examples.

Public metrics count **flows inheriting session decisions**. Longer sessions contribute more examples. They are not session-level scores or independent per-flow decisions. The raw browser includes all flows, while the report's `sample_flows` and `alerts.sample` are bounded samples. Raw rows contain generated labels, not per-flow model predictions.

Saved sequence artifact directory contains `model.pt`, `preprocessor.pkl`, and `metadata.json`. Metadata records architecture, class order, hidden size, and maximum length; this older artifact lacks some fields now written by the current exporter. Use the saved preprocessor and compatible model construction. Only load trusted local pickle/model artifacts.

## 8. Model progression and timing evidence

`ui/versions.json` preserves historical comparison output. `ui/research.json` supplies the current main four-model table via `four_model_comparison`, plus archived studies that are not all rendered. `public_model_replaced` is false.

The four-model follow-up uses exact saved parent sessions for seeds 701, 733, and 769. Beta 2/Sona reuse relative-pace candidates; Beta 1/Beta 3 are newly trained counterparts with narrower ordinary training. All receive shared timing pairs and harmless polling. This is an extension on known cases, not fresh independent confirmation or a retest of original public weights.

| Candidate | Ordinary accuracy | Lower-rate accuracy | Timing accuracy | Equal three-test mean |
| --- | ---: | ---: | ---: | ---: |
| Beta 1 | 100% | 100% | 50% | 83.33% |
| Beta 2 | 100% | 100% | 50% | 83.33% |
| Beta 3 | 100% | 91.67% | 99.19% | 96.95% |
| Sona candidate | 100% | 99.24% | 99.00% | 99.41% |

There are 131 ordinary, 132 lower-rate, and 3,600 timing test sessions across the three seeds. The scores are means across seeds. The three-test score averages category means equally, rather than pooling all sessions. Sona's lower-rate result corresponds to one error among 132 sessions. Beta 3 is slightly better on familiar timing; Sona has the strongest specified three-category mean.

Timing pairs have identical feature multisets, length, and endpoints but different interior ordering. One member has cyclic timing and the other shuffled ordering. Order-invariant per-flow majority voting must give both the same answer, yielding 50% on balanced pairs. A GRU can distinguish the sequence. Shuffling controls and artifact reload audits support this interpretation. This establishes a capability on deliberately constructed examples, not superiority over every deterministic detector. Engineered temporal rules or statistical sequence features can also use order.

Training recipes differ beyond architecture: MLP counterparts use up to 80 epochs, patience 10, batch 512, inverse-frequency weighting; GRU counterparts use up to 200 epochs, minimum 60, patience 20, batch 128, weight power 0.5. Do not attribute every score difference solely to memory. Lower-rate attacks retain per-flow clues, explaining why simpler models can excel there. An exact cause for an individual GRU error requires case inspection.

Stress categories include altered timing pace, longer cycles, subtle contrast, stronger noise, and harmless polling. Performance weakens on subtle/extreme-noise cases. No inference-speed benchmark was performed. Larger timing experiments do not increase the public raw browser's row count or silently retrain its checkpoint.

Detailed protocols: `docs/FOUR_MODEL_COMPARISON.md`, `docs/TEMPORAL_BENCHMARK.md`, `docs/TEMPORAL_ROBUSTNESS.md`, and `docs/PACE_NOISE_STUDY.md`. Generated audit/model/prediction files under `data/generated/` are ignored and may be absent in a clone. A general summary document is not a substitute for those files when reproducing results.

### Mixed-traffic gradual-scan follow-up

A separate prospective benchmark is implemented in `experiments/mixed_scan.py`, with independent reload auditing in `audit_mixed_scan.py` and all-seed summaries in `summarize_mixed_scan.py`. Its protocol is documented in `docs/MIXED_SCAN_BENCHMARK.md`. Unlike the earlier paired-order experiment, it independently generates gradual scanning amid background traffic and benign retries, monitoring, discovery, and inventory activity. Observation windows contain mixed event-level scenarios, so it uses a separate `window_target` and its own tensor packer; the public single-scenario session builder is unchanged.

Seeds 811/827/853 each use 2,400 train, 1,600 validation, and 2,400 windows in each of three test categories. The primary comparison freezes thresholds at a validation false-positive budget of 1%, then reports both scan recall and actual test false-positive rates. Competitors include connection voting/mean scores, an order-independent mean-pooled neural model, summary boosting, a temporal rule, and shuffled GRU controls. The flow classifier repeats window targets as weak flow labels; the pooled model receives direct window-level supervision. No learner receives event-level generator labels or behavior names as features.

These are binary architecture candidates, not new public six-class checkpoints. Protocol, raw data, predictions, source hashes, models, thresholds, and audits are saved under ignored `data/generated/mixed_scan_v1/`. Results must be read from the completed audited report; the test does not guarantee a GRU win or a 1% test false-positive rate. Legitimate inventory can resemble scanning, and the experiment does not establish online detection latency or malicious intent from flow records.

The completed mixed-scan run generated 3,167,532 flows / 33,600 windows. Familiar scan recall / actual false-positive rate: connection voting 3.58% / 0.89%, mean-pooled MLP 3.25% / 0.83%, Sona-style GRU 20.14% / 0.78%, summary boosting 28.83% / 1.56%. Sona reaches 10.50% / 0.58% in longer changed windows and 9.67% / 1.28% with more legitimate inventory. Shuffled-test and shuffled-trained familiar recalls are 1.22% and 4.86%. These support useful learned context relative to connection voting, but low recall, inventory ambiguity, and mismatched realized false-alarm rates prevent a universal superiority or promotion claim. Full results and next-study recommendations are in the benchmark document; public Overview and comparison remain unchanged.

### Causal context and pace follow-up

`experiments/scan_context.py` derives causal timing and history features: pace relative to earlier positive gaps, earlier failure and port-diversity counts, recent 8/32-event summaries, past-300-second failures, and current-port novelty. An observer cutoff enforces nominal capture bounds uniformly, addressing a few late inventory events in the older simulator without changing its archived source or outputs. Prefix/provenance tests verify that future observations and labels do not alter earlier features.

`mixed_scan_context.py` freezes fresh seeds 941/967/983 with the same train/validation/test sizes and generator distributions as the first study. It compares raw, pace-aware and full-context 32-unit GRUs; context pooling/voting/mean scores; original/context summary boosting; a temporal rule; and shuffled context-GRU evaluation. Context comparators already receive engineered history, so they are not history-free even if their networks are non-recurrent. Validation budgets 1%, 2%, and 5% are all retained; actual test false alarms are reported separately. Extra inputs increase input-layer parameters, with hidden size unchanged.

`audit_scan_context.py` independently reconstructs every candidate and budget, verifies capture bounds, identity/source hashes and train-only preprocessing, and reproduces saved scores/metrics. `summarize_scan_context.py` requires a passed audit and retains all seeds and operating points. Artifacts are isolated under `data/generated/mixed_scan_context_v1/`; an interrupted pretest run is archived under `mixed_scan_context_preflight_v1/`. These benchmark reports are not compatible public dashboard exports and must not be copied into `ui/report.json`. Operational role/schedule context is not implemented, and no oracle authorization flag is used. Protocol and outcomes are documented in `docs/SCAN_CONTEXT_BENCHMARK.md`.

The completed context follow-up generated 3,172,119 fresh flows / 33,600 windows. At the 1% validation budget, familiar recall / actual false-positive rate is raw GRU 29.44% / 1.08%, pace-aware GRU 45.47% / 1.25%, full-context GRU 24.69% / 1.33%, and context pooling 34.11% / 1.69%. At the 5% budget, pace-aware familiar recall rises to 78.19% with 5.08% actual false positives: this is a threshold tradeoff, not a model upgrade. On longer changed windows, raw recall is 13.83% versus pace-aware 1.69%; on the inventory-heavy test it is 13.81% versus 1.72%. The full context fields do not fix these weaknesses. No candidate was promoted. All GRUs retain hidden size 32; input changes produce 5,250 / 5,442 / 6,690 parameters for raw / pace / context. Next recommendations are independent duration/density training variation, representative benign calibration, a relative-only timing ablation, and fresh confirmation, not a larger network or an oracle intent feature.

### Independent duration and traffic-volume follow-up

`mixed_scan_generalization.py` retrains raw, relative-only timing and small-history GRUs, matched small-history pooling, summary boosting and the temporal heuristic. `scan_generalization_features.py` creates four independent duration/background-count cells and uses scale-invariant causal startup timing: relative timing is zero until an earlier positive gap establishes a reference. Relative representations remove `prior_gap_s` but retain the other original flow inputs. The small subset adds three recent-history/port-novelty features. No labels or generator intent enter inputs.

Fresh seeds 1103/1129/1151 each have 3,200 mixed-condition training windows, 3,200 mixed validation windows and five independent test partitions of 1,600 windows. Durations are 30/60 minutes; background ranges are 48 to 84 or 64 to 116 connections plus identically distributed probe counts. Training includes every ordinary test cell, so this tests fresh-sample performance within those conditions, not unseen-duration extrapolation. A fifth test increases benign inventory prevalence. No ordinary-only training control isolates augmentation.

Completed training generated 4,202,045 flows / 43,200 windows. Strict-budget raw versus relative mean recall / actual false alarms: familiar 28.04% / 0.96% versus 49.46% / 1.38%; duration-only 40.50% / 1.50% versus 50.17% / 1.50%; background-only 27.25% / 1.00% versus 52.33% / 1.79%; combined 41.25% / 1.71% versus 51.00% / 1.71%; inventory-heavy 42.71% / 5.00% versus 52.62% / 5.21%. Relative timing improves all condition means but not every seed. Small-history GRU/pooling do not improve the raw control. Hidden size remains 32; raw/relative/small GRU parameters are 5,250 / 5,346 / 5,634. About half of scan windows remain missed at the strict setting, and legitimate inventory causes substantial false alarms.

For these experimental artifacts, always call `prepare(frame)` from `scan_generalization_features.py` and pass it explicitly to `processor.transform(frame, prepared)` or `pack_context(frame, processor, cap, prepared)`. The reused low-level `ContextPreprocessor` defaults to the older context calculator if `prepared` is omitted, which does not implement the new scale-free startup. The runner and auditor both supply the correct prepared rows. Do not use the omitted-argument path for generalization inference.

`audit_scan_generalization.py` replays every candidate and threshold; `summarize_scan_generalization.py` requires passed auditing and exports all results plus per-cell validation false alarms. Local artifacts are `data/generated/mixed_scan_generalization_v1/`; the report remains an experimental binary-window contract, incompatible with the public six-category dashboard. All 119 backend tests passed; all three full audits passed 108 prediction replays and 324 operating-point checks, plus frozen implementation and public/prior artifact preservation hashes. Complete all-candidate tables and calibration-cell records are saved locally alongside the audit. See `docs/SCAN_GENERALIZATION_BENCHMARK.md` for protocol, complete interpretation and verification updates. Next work should isolate augmentation and relative-memory contributions and test unseen durations before any six-category integration or promotion.

### Advanced candidate and independent evaluation

The isolated `experiments/sona_candidate/` package compares causal relative timing, ordinary/mixed training, small and capacity-matched pooling, hierarchical memory, causal retrieval, and GRU capacity controls. Across three seeds it generated 5,043,301 fresh flows / 54,000 windows and trained 36 full models. Validation selected a relative-timing GRU40; hierarchical/retrieval designs did not establish an improvement. This remains a binary-window experiment, not a six-category public replacement. Optimized preprocessing reproduced exact features and was 4.77–5.83× faster in paired measurements; that is not an end-to-end model speed claim.

`experiments/sona_evaluation/` supplies an independent, frozen suite with shifted traffic, memory controls, saved-checkpoint transfer and retrained counterparts. The primary three-seed sealed comparison judged GRU40 against retrained GRU32 **INSUFFICIENT EVIDENCE**: mean recall difference −0.15 percentage points, 95% interval [−0.49, +0.21]. A separate three-seed replication gave **FAIL**: −0.67 points [−0.96, −0.37]. Long-sparse false alarms reached 24.58% in the primary run versus 4.67% for GRU32. Both GRUs demonstrated memory, but GRU32 already solved the suite’s chain task; the candidate did not establish deeper reasoning or overall superiority.

Retrained GRU32 ranks familiar mixed-window traffic better than the Beta counterparts (mean ROC AUC 0.997 versus 0.937/0.920). The public saved Sona checkpoint transfers poorly to that new window family; do not substitute retrained architectural results for delivered checkpoint capability. Beta voting thresholds interact with benign chain examples in validation and exaggerate their recall deficit at the primary operating point. Preserve that limitation rather than presenting their near-zero recall as a universal comparison.

The independent efficiency reference workload measured candidate end-to-end latency at 46.78 ms versus retrained GRU32 at 42.11 ms and Beta 2 at 25.63 ms. All measurements flagged concurrent CPU load. The candidate is not universally faster. The completed candidate backend check recorded 164 passing tests; the candidate did not establish independent acceptance, and six-category promotion was not attempted. Full protocols, all outcomes, artifact contracts, and limitations: `docs/SONA_CANDIDATE.md` and `docs/SONA_EVALUATION.md`.

No experimental checkpoint replaced Overview. Generated weights/predictions remain under ignored `data/generated/` paths. Future work must use fresh development and sealed cases, reliable per-condition false-alarm gates, and a separate six-category regression before promotion.

## 9. Static frontend

Five pages: `index.html`, `model.html`, `progression.html`, `neurons.html`, and `raw.html`. Shared CSS is `styles.css`; page logic lives in `app.js`, `model.js`, `versions.js`, `neurons.js`, and `raw.js`. Contract/helpers are `data.mjs`, `raw-data.mjs`, `versions-data.mjs`, and `research-data.mjs`.

Overview order: detection performance, classification/dataset summary, detection records, alerts/incidents. Investigation uses inline detail panels, with normal traffic and benign false positives distinguished from actual attacks. Sona explained uses an illustrative neural SVG and selectable stages, not measured neuron activations. Sona Progression uses the matched candidate comparison; Beta 1/2 share a row when their three means match, while their individual training descriptions remain inspectable. Neural networks explained uses a separate 2→3→2→2 teaching network with individually selectable neurons, editable inputs/connection weights, actual responses and class percentages. The short learning section explains backpropagation; no training controls or symbolic equations are exposed. Its math helper retains tested gradient routines, but the page uses only inference. No public report import/export controls remain.

`mobile.mjs` installs delegated metric help for dynamically rendered cards: tap toggle, desktop hover, keyboard focus, Escape/outside-tap/scroll/resize dismissal, and viewport-aware positioning. CSS provides safe-area handling, a two-column phone layout for the four workspace tabs plus a full-width dataset action, coarse-pointer touch targets and 16px form text, swipeable tables/diagram, stacked panels, and landscape rules. On phones the header scrolls away, compact classification/progression tables fit their cards, and wide record tables/diagrams have visible swipe cues and a draggable horizontal position control. Neural input/output flows use short single-row labels; the ten-step selector uses a compact two-row grid. Automated checks do not establish physical iPhone Safari appearance.

Raw data is stored as row arrays in numbered JSON chunks plus `raw-data/manifest.json` listing columns and chunks. The current bundle has 88 chunks and 87,505 records. The page loads listed chunks and searches the complete dataset, not only the current page. Default pagination is 100 rows. Report/manifest run, count, and generation provenance must agree.

The report validator accepts versions 1.0.0 and 1.1.0. The latter supports session-detector metadata. Preserve finite-rate/count checks, matrix consistency, class ordering, and sample-versus-total distinctions. Never invent missing metrics or silently reinterpret an unsupported contract.

## 10. Refresh and release workflow

Read-only preview and checks:

```bash
npm --prefix ui run dev
npm --prefix ui run check
.venv/bin/python -m pytest
```

The Node checks include contract/filter tests, lightweight DOM interaction tests, raw-data integrity, mobile-help behavior, and standalone packaging checks. The DOM harness is not a visual browser. Backend tests exercise generation, schema, training, sequence handling, alerts, replay, reports, and experiment audits. Run checks appropriate to a change rather than retraining large experiments for a documentation edit.

Refreshing presentation data is an explicit mutation, separate from training:

```bash
npm --prefix ui run sync-report
npm --prefix ui run sync-raw
npm --prefix ui run check
```

Default report source is `data/generated/showcase_v4/report.json`. `sync-report` accepts an alternate report path. The raw exporter follows `artifact_paths.dataset` unless given an explicit dataset directory. Refresh report and raw data together; updating only one can invalidate the dashboard. These commands assume the required local generated artifacts and Python environment exist.

For the experimental comparison only, from the repository root:

```bash
.venv/bin/python ui/scripts/sync-research.py \
  data/generated/sona_pace_noise_v1 \
  --four-model-summary data/generated/four_model_timing_v1/summary.json
```

Inspect the exporter and source summaries before using it. This does not promote the experimental Sona model into Overview. Promotion requires a deliberate checkpoint/preprocessor choice, a compatible public dataset/evaluation/report export, audit checks, and coordinated UI refresh. Do not simply copy candidate percentages into the old report.

Standalone release tooling is `ui/scripts/build-release.mjs`. An explicitly requested build validates data, embeds all five pages/scripts/styles and JSON including every raw chunk into `Download_Project_Here.html`, updates `CURRENT_RELEASE.txt`, and preserves older numbered snapshots. Subsequent builds refresh the same named download. Navigation uses a parent iframe and URL fragments; fetches resolve to embedded resources. Multi-file modules require HTTP, while the standalone file supports offline use with JavaScript enabled.

**The current downloadable release is `Download_Project_Here.html`.** Build command, only when requested: `npm --prefix ui run build`. The current standalone file is about 20.9 MiB because it contains the full dataset.

Multi-file hosting configuration lives in `ui/vercel.json`: static files, no server functions, no production ML process. The single-file release uses inline scripts and styles; hosting policy must permit those rather than blindly applying the multi-file site's external-script restrictions. No build or deployment is implied by editing docs.

## 11. Practical constraints for future work

Keep public wording clear, without emojis, em dashes, or implied team ownership. Public names are Sona and Beta 1/2/3. Internal identifiers remain v1/v2/v3/v4. Ground truth must never be presented as a model discovery. Confidence-based severity is not independently assessed danger.

Do not mix historical macro F1, candidate session accuracy, public flow-counted metrics, or category means under one unlabeled percentage. Do not describe Sona as faster, universally superior, trained on captured production traffic, or able to identify software vulnerabilities.

The static dashboard has no automatic refresh from backend training. Every displayed result must be traceable to the correct saved artifact. Preserve closed-session assumptions and label exclusion when adding timing features. Validate session/run identity when combining frames. Extending `SCENARIOS` changes class/output compatibility and requires deliberate schema/artifact handling.

Source and current artifact contents outrank old prose. Several older guides describe the earlier two-network workbench and historical tests. This file and the root README distinguish the current public path from those still-supported workflows. Do not delete older backend modules simply because they are not the public deployment path.

Local environments, generated backend artifacts, credentials, agent coordination files, and local maintenance tools are excluded by `.gitignore`. Public technical documentation should not depend on local agent notes to make sense. Before changing code, inspect repository instructions if present, relevant modules, artifact metadata, and working-tree changes. Never rebuild releases, retrain models, install dependencies, commit, push, or deploy merely because this reference lists the commands.

## 12. Source publication boundaries

Python implementation and tests under `src/`, `tests/`, and `ui/scripts/` are tracked source files, not excluded by `.gitignore`. At this documentation update, all 114 project Python files are in the commit published to GitHub’s `main`. The remote is `https://github.com/khicklin23/Sona.git`. Counts will change as source is added.

Ignored local material includes `.venv/`, Python caches, `data/generated/`, `artifacts/`, secrets, Git-kb/agent directories, `AGENTS.md`, and local coordination documents. A clone includes source and bundled static UI data; it does not include the training environment or saved backend weights/datasets. The standalone download embeds the five dashboard pages and full showcase raw data, so it can be viewed without those backend artifacts.
