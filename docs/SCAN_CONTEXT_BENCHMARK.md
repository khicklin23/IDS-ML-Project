# Causal context and traffic-pace follow-up

## Frozen question

Does observable history improve mixed-traffic scan detection, and how much of the benefit requires a GRU rather than an order-independent learner with those same inputs?

This follows the difficult first mixed-scan study. Its old test cases remain frozen. New seeds are **941, 967, 983**. For each seed: 2,400 training windows, 1,600 validation windows, and 2,400 windows each for familiar traffic, longer windows/changed background, and more legitimate inventory. The behavioral distributions are unchanged; fresh seeds create independent environments and identities. A uniform observer cutoff excludes any event outside its configured window, correcting occasional inventory events that the original simulator schedules past the nominal end. The original generator and earlier results are preserved. It is a new sample from the same simulator family, not a captured-traffic validation.

Every candidate uses the same window targets. Training remains 30-minute windows; longer 60-minute conditions are reserved for stress testing, so training augmentation does not confound the representation comparison. Validation selects separate thresholds for **1%, 2%, and 5%** benign flag budgets. All three points are retained and frozen for every test. Actual test false-alarm rates can exceed those budgets.

## Inputs and comparators

The original flow inputs remain available. Added timing features divide the current gap and elapsed time by the mean positive gap from **earlier** observations, then apply `log1p`. The initial fallback is a fixed 30 seconds. There is no complete-window normalization or use of future observations.

Added history features count earlier flows, failed connections, distinct ports and distinct failed ports. They also describe the last 8 and 32 observations, failures in the past 300 seconds, and whether the current destination port has appeared earlier. Counters reset at the supplied observation-window boundary. These are deterministic telemetry calculations, not learned weights or attack labels.

| Candidate | Input / role |
| --- | --- |
| Raw Sona-style GRU | Original inputs; retrained control on the same fresh data |
| Pace-aware GRU | Original inputs plus two causal timing features |
| Context-aware GRU | Original inputs, timing and history features |
| Context neural pooling | Identical context inputs with directly supervised mean pooling |
| Context connection voting / mean | Identical context inputs, repeated window labels as weak flow supervision |
| Original summary boosting | Existing full-window summaries, same fresh observations |
| Context summary boosting | Original summaries plus mean/max/final causal context values |
| Context GRU shuffled evaluation | Shuffle encoded vectors after context extraction; diagnostic only |
| Temporal rule | Same original temporal heuristic, validation-selected cutoff |

Context inputs already encode earlier activity. Therefore context pooling/voting are not history-free detectors, even though their networks are not recurrent. A context-GRU improvement over the raw GRU is not automatically evidence that recurrent memory caused it. The matched pooling and boosting controls address this distinction.

All GRUs keep 32 hidden units. Maximum 80 epochs, minimum 20, patience 12, batch 128, Adam learning rate 0.001, unweighted window cross-entropy. Pooling uses the same training schedule. Connection MLP uses maximum 80 epochs, patience 12, batch 1,024, and its standard class weighting. Both boosting baselines use 150 iterations, 15 leaves, L2 regularization 1. Checkpoint selection uses validation loss. Maximum sequence length 160; oversized windows raise an error. Extra inputs increase input-layer parameter counts, which are recorded, but hidden capacity is not increased.

## Inspection and limits

Before designing this follow-up, the first study's familiar test windows were inspected by predicted outcome and benign behavior. Legitimate inventory caused 21 of 28 Sona false alarms; 2,875 of 3,600 scan windows were missed at the strict 1% validation budget. The inspection CSV records port diversity, failure behavior, probe span, and gap variation for every old test window. It is exploratory analysis, not a fresh confirmation set.

No actor intent, generator behavior name, event label, window target, IP value, role oracle, or authorization flag becomes a feature. The study does not yet add asset-role or schedule metadata: the old generator has no independently modeled operational context for those fields. Inventing an authorization feature derived from the target would make an invalid benchmark.

Feature causality does not mean the final alert is online: predictions still use complete windows. Prefix alert timing remains a future evaluation. Benign inventory and scanning can remain observationally ambiguous. Balanced prevalence does not establish production alert precision. No candidate is assumed to win, and all results are retained.

## Reproduce and verify

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.mixed_scan_context \
  --out-dir data/generated/mixed_scan_context_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.audit_scan_context \
  data/generated/mixed_scan_context_v1
```

The output directory must be new. Protocol and implementation hashes are saved before generation/training. Each seed saves raw Parquet partitions, all candidate weights, three fitted preprocessors, both boosting models, training histories, prediction arrays, frozen thresholds, parameter counts and results. Auditing reconstructs models, refits train-only preprocessing, reproduces scores and all three operating points, and checks identities/source hashes and aggregate consistency.

Public Sona/checkpoints, dashboard data, prior experiment outputs and HTML releases remain unchanged. No real-network capture or external dataset download is part of this study. The initial preflight was stopped before test evaluation when a boundary check found a handful of late inventory observations. Its files remain in `data/generated/mixed_scan_context_preflight_v1`. The corrected run applies the same observation cutoff to every label and activity type, then recomputes gaps. No seed, budget, architecture, or behavioral distribution was changed in response to test results. Verified results and interpretation are recorded below.

## Completed training results

All three fresh seeds generated **3,172,119 flow records in 33,600 observation windows**. Each test category contains 7,200 windows across seeds, half scan-containing and half benign. Means below compare candidates on identical fresh observations; the first study's older scores are not the control.

### Familiar traffic and false-alarm tradeoffs

| Validation false-alarm budget | Candidate | Scan recall | Actual test false-positive rate |
| --- | --- | ---: | ---: |
| 1% | Raw GRU | 29.44% | 1.08% |
| 1% | Pace-aware GRU | 45.47% | 1.25% |
| 1% | Full-context GRU | 24.69% | 1.33% |
| 1% | Context neural pooling | 34.11% | 1.69% |
| 1% | Context summary boosting | 24.94% | 0.89% |
| 2% | Raw GRU | 43.19% | 1.83% |
| 2% | Pace-aware GRU | 59.39% | 2.14% |
| 2% | Full-context GRU | 41.53% | 2.39% |
| 5% | Raw GRU | 68.25% | 4.89% |
| 5% | Pace-aware GRU | 78.19% | 5.08% |
| 5% | Full-context GRU | 65.25% | 4.67% |

Raising the budget changes the operating threshold, not the trained model. Better recall at 5% is purchased with more false alarms. Equal validation budgets do not force equal test rates.

### Changed conditions at the strict setting

| Test | Candidate | Scan recall | Actual test false-positive rate |
| --- | --- | ---: | ---: |
| Longer windows / changed background | Raw GRU | 13.83% | 0.72% |
| Longer windows / changed background | Pace-aware GRU | 1.69% | 0.03% |
| Longer windows / changed background | Full-context GRU | 8.06% | 0.44% |
| Longer windows / changed background | Context neural pooling | 2.19% | 0.03% |
| Longer windows / changed background | Context summary boosting | 5.11% | 0.08% |
| More legitimate inventory | Raw GRU | 13.81% | 1.25% |
| More legitimate inventory | Pace-aware GRU | 1.72% | 0.03% |
| More legitimate inventory | Full-context GRU | 7.17% | 1.36% |
| More legitimate inventory | Context neural pooling | 2.36% | 0.31% |
| More legitimate inventory | Context summary boosting | 5.19% | 1.03% |

Low false-alarm rates from a model that misses almost every scan are not a detection success. The longer-window tests change elapsed time and background flow count together; these scores do not isolate the causal reason for the failure.

## Interpretation

Pace features improved average familiar recall by 16.03 percentage points, at a 0.17-point increase in actual false-positive rate. The improvement is not universal across seeds: raw familiar recall ranged from 20.00% to 44.75%, and pace-aware recall from 42.67% to 50.83%; one seed favored the raw model. This is useful evidence of a representation benefit in familiar conditions, not proof of a robust upgrade.

The full set of history features did not improve the GRU over the raw control. Context neural pooling uses the same deterministic history and can perform competitively, showing that supplying history and learning recurrent memory are different contributions. Extra measurements or parameters do not guarantee a stronger detector.

Legitimate inventory remains a major ambiguity. At the familiar strict setting, the raw GRU flags 20 of 176 benign inventory windows; the pace-aware model flags 22; the full-context model flags 37; and context pooling flags 46. These are 11.36%, 12.50%, 21.02%, and 26.14% within that benign subtype. Their overall false-positive rates are smaller because inventory is only a minority of benign windows.

All GRUs have 32 hidden units. Actual parameter counts are raw 5,250, pace-aware 5,442, and full-context 6,690. Context pooling and voting have 1,714 parameters each. The extra parameters come from new input projections, not wider memory. Those increases did not solve observational ambiguity or unfamiliar-condition performance.

The pace-aware model retains absolute gap measurements alongside the relative ones, so adding relative features does not enforce pace invariance. Training covers only 30-minute windows with the usual background-volume range, while stress tests use 60-minute windows with more background records. The observed failure is consistent with sensitivity to changed inputs and traffic composition, but needs isolated tests to establish a specific mechanism. More context alone did not fix it.

## Next direction

Keep all current artifacts frozen and do not promote these candidates. The next predeclared study should vary observation duration and background density independently during training, calibrate on representative benign conditions, and reserve fresh seeds plus separate timing-only and density-only tests. A relative-only timing ablation can test whether retaining absolute gaps encourages an unreliable shortcut. Retain a small context subset and strong pooling/summary competitors; do not assume all thirteen history fields help.

The 1% budget needs enough benign validation windows to estimate rare errors reliably. Legitimate inventory should be reported separately. Independently observed job/asset context may eventually help distinguish authorized scanning, but deriving an authorization feature directly from the target would invalidate the experiment. No role/schedule oracle was added here.

Complete all-candidate, all-budget means/ranges and per-seed records are written locally to `data/generated/mixed_scan_context_v1/RESULTS.md` and `per_seed.csv` after independent auditing. No overall winner score combines familiar success with stress failure.

## Verification

All three independent audits passed: 120 model/split prediction checks and 360 operating-point checks reproduced validation-selected thresholds, metrics and behavior-specific errors. Capture bounds, disjoint identities, source hashes, train-only preprocessing and aggregate consistency were verified. The complete corrected backend suite passed 116 tests; five dedicated context tests cover prefix/provenance independence, timing expiry, preprocessing, no truncation, full artifact replay/all-budget summaries and capture bounds. Public checkpoint/data/release hashes and prior mixed-scan report/audit remain unchanged. No installs, commits, pushes, deployments or new HTML releases were performed.
