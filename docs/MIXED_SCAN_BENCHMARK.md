# Gradual scanning amid normal traffic

## Question and frozen protocol

Can a Sona-style GRU detect gradual scanning mixed with ordinary host activity at a validation false-alarm budget of 1%, and does it beat strong methods that do not use learned sequence memory?

Protocol fixed before the full run on 2026-10-12: seeds 811, 827, 853; 2,400 training windows, 1,600 validation windows and 2,400 windows per test category per seed. Classes are balanced for this capability study. Every window independently generates background traffic plus probe-like connections; there are no rearranged pairs or enforced identical feature multisets.

Training/validation/familiar tests use 30-minute windows, 48–84 background flows and 8–28 probe-like flows. Changed-environment tests use 60-minute windows with 64–116 background flows. Attack windows contain gradual probes across a sampled set of service ports, spread across most of the window. Benign windows include ordinary activity, retries, monitoring, bursty service discovery and occasional longer inventory activity. The hard-lookalike test increases legitimate inventory from 5% to 25% of benign windows. Counts of probe-like events have the same distribution for attacks and normal activity. Their connection measurements overlap; no one category owns a unique protocol or outcome. Source environments/hosts/windows are disjoint between splits.

Each flow retains event-level generator ground truth for auditing, but all learners receive only the same **window target** and the allowed telemetry fields. The window label means a generated scan is present among background connections; it is not a claim that every connection in the window is malicious. This is separate from the public six-class session builder, which assumes one scenario per session.

## Comparators

- **Beta 2-style connection voting:** retrain the existing small connection MLP with the window target repeated across its constituent flows. Report voting and mean probability. This weak supervision is imperfect for mixed windows; it is not a retest of the original Beta 2 checkpoint.
- **Mean-pooled MLP:** the same connection architecture trained directly against the window label through mean pooling of connection logits. This is an order-independent neural control with better-matched supervision than repeated flow labels.
- **Sona-style GRU:** the existing 32-unit architecture retrained for binary window classification. No public weights are changed.
- **Summary boosting:** histogram gradient boosting on engineered counts, port diversity, failure fractions, probe spans/gaps, activity bins and size/duration summaries.
- **Temporal rule:** diverse failed short attempts sustained across 300-second bins. Its threshold is selected on validation, rather than intentionally choosing a weak cutoff.
- **Order controls:** evaluate Sona on shuffled flow vectors and train a second GRU with shuffled sequences. These change observation order for ablation only; the primary test data remains chronological and independently generated.

Allowed sequence inputs remain the schema's eleven columns. Identity and provenance are excluded. Summary/rule features use those measurements and elapsed time, which can be reconstructed from gaps. Their chronological statistics deliberately give non-GRU methods useful temporal information.

GRU and pooled model: maximum 80 epochs, minimum 20, patience 12, batch size 128, learning rate 0.001, hidden size 32. GRU loss is unweighted cross-entropy because window classes are balanced. Connection MLP: maximum 80 epochs, patience 12, batch size 1,024, default inverse-frequency class weighting. Summary boosting: 150 maximum iterations, 15 leaves, L2 regularization 1. Checkpoint selection uses validation loss; test results do not select a seed or architecture. Maximum sequence length is fixed at 160; exceeding it raises an error rather than truncating evidence.

## Measurement

For each learner, validation selects the lowest score threshold whose benign flag rate is at most 1%, with ties handled conservatively. That threshold is frozen for every test category. Report attack recall and actual test false-positive rate separately: the validation budget does not guarantee a 1% test rate, especially under distribution shifts. Also retain ROC AUC, average precision, default probability metrics where meaningful, and false flags by benign behavior.

No single combined score is used to declare Sona superior. The key evidence is whether recall improves without a materially higher test false-alarm rate, across seeds and conditions. All seeds and controls must be retained regardless of outcome.

## Reproduce

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.mixed_scan \
  --out-dir data/generated/mixed_scan_v1
```

The output directory must be new. `protocol.json` is written before generation/training; each seed saves raw Parquet splits, weights, preprocessing, training histories, validation/test predictions, thresholds, source hashes and metrics. `checkpoint.json` preserves completed runs. `report.json` combines means and ranges across seeds. Generated outputs remain local and ignored by Git.

## Limits

This is still a behavioral simulator, not captured real traffic. Intent labels can disagree with what flow telemetry can establish: legitimate inventory can resemble scanning. The observations are complete, non-overlapping windows with supplied source grouping, not live discovery of attack sessions. Fixed windows do not demonstrate detection delay; that requires a future causal prefix evaluation. Balanced test prevalence does not estimate operational alert precision on a mostly-benign network. Mean-pooled training, summary features, and the rule make strong competing methods available; a GRU win is not assumed.

Public Overview/report/raw data, historical comparisons, UI and standalone HTML releases are unchanged. Verified results and interpretation are recorded below.

Independent replay and summary commands after training:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.audit_mixed_scan data/generated/mixed_scan_v1
.venv/bin/python -m ids_ml.experiments.summarize_mixed_scan data/generated/mixed_scan_v1
```

The audit reconstructs models from saved weights, refits preprocessing on the saved training partition for comparison, reproduces every validation/test score, reselects validation thresholds, checks split identities and source hashes, and recomputes metrics and behavior-specific false alarms. The summary retains every model and seed in `RESULTS.md` and `per_seed.csv`.

## Completed frozen-run results

The three runs generated **3,167,532 flows in 33,600 observation windows**. Each test category contains 7,200 windows across the three seeds, half scan-containing and half benign. The table gives means across seeds at each model's validation-frozen threshold. Test false-positive rates are measured outcomes, not forced to 1%.

| Test | Model | Scan recall | Actual false-positive rate |
| --- | --- | ---: | ---: |
| Familiar mixed traffic | Connection voting | 3.58% | 0.89% |
| Familiar mixed traffic | Mean-pooled neural model | 3.25% | 0.83% |
| Familiar mixed traffic | Sona-style GRU | 20.14% | 0.78% |
| Familiar mixed traffic | Summary boosting | 28.83% | 1.56% |
| Longer windows / changed background | Connection voting | 0.00% | 0.00% |
| Longer windows / changed background | Mean-pooled neural model | 0.08% | 0.03% |
| Longer windows / changed background | Sona-style GRU | 10.50% | 0.58% |
| Longer windows / changed background | Summary boosting | 7.39% | 0.22% |
| More legitimate inventory | Connection voting | 0.03% | 0.00% |
| More legitimate inventory | Mean-pooled neural model | 0.17% | 0.03% |
| More legitimate inventory | Sona-style GRU | 9.67% | 1.28% |
| More legitimate inventory | Summary boosting | 7.06% | 1.11% |

### What this supports

Sona provides useful additional context relative to individual-connection voting and order-independent neural pooling on this independently generated mixed-traffic task. On familiar traffic it detects 725 of 3,600 scan windows, versus connection voting's 129, while producing 28 false flags versus voting's 32. Its recall varies from 16.42% to 25.75% across seeds. This is a more practical capability comparison than an intentionally indistinguishable pair test, but it is still synthetic.

Useful order dependence is supported by the controls: familiar recall drops to 1.22% when Sona's observation vectors are shuffled, and the shuffled-trained GRU reaches only 4.86%. Their actual false-positive rates are 0.86% and 1.19%. These controls share the same vector measurements; they do not simulate realistic delayed logging. Training settings and model capacities differ between MLP and GRU, so do not attribute every difference exclusively to memory.

Sona is not a universal winner. Summary boosting catches more familiar scans, at a higher test false-alarm rate. Its mean ROC AUC is 0.950, versus Sona's 0.938, showing stronger overall score separation on familiar traffic. ROC AUC measures ranking across thresholds and is not the attack recall at the fixed low-false-alarm operating point. On longer-window and inventory tests Sona catches more scans than summary boosting, but also has more false alarms. These tradeoffs do not establish superiority at identical realized test false-positive rates.

### What still fails

Even Sona misses almost 80% of familiar scan windows at this strict operating point, and more under changed conditions. A near-perfect score is not supported. Longer windows change timing scale and background simultaneously, so this stress test does not isolate the exact cause of performance loss.

Legitimate inventory is the main familiar false-alarm source: 21 of Sona's 28 false flags come from 171 inventory windows, a 12.28% flag rate within that benign subtype. Summary boosting flags 42 of those inventory windows. Flow telemetry cannot reliably distinguish authorization or intent when legitimate inventory and scanning produce similar observations. Increasing neural capacity cannot create missing information.

### Next experiment

Keep these results frozen. A subsequent study should predeclare a hybrid model with causal rolling port-diversity/failure summaries and pace-normalized timing, exposing the same additional inputs to strong competing methods. It should also test legitimately observable context such as asset role and scheduled activity, without passing generator behavior labels or an oracle authorization flag. Use fresh held-out seeds/environments for confirmation, include legitimate scanners, and retain the strict false-alarm budget. This separates representation improvements from more parameters and avoids tuning repeatedly against this test set.

Do not promote these binary candidates into the public six-class model. The current evidence supports a useful memory capability with substantial detection and calibration limits; deployment-quality robustness is not established.

## Verification

All three independent seed audits passed, reproducing 96 model/split evaluations, validation thresholds, per-behavior false alarms, source hashes, and disjoint identities. The complete backend suite passed 110 tests; six dedicated experiment tests cover mixed observations, provenance exclusion, padding/order controls, operating points, artifact replay, and summary preservation. The public report/comparison/raw manifest and existing Release_1.15.html remain byte-for-byte unchanged. No installations, commits, pushes, deployments, or new HTML releases were performed.

A later preflight for the context follow-up found that the simulator occasionally schedules a few legitimate inventory events beyond a nominal window end. The original archived data and scores remain unchanged; these were supplied logical observation groups, with this small timing-boundary limitation. The context follow-up applies a uniform capture cutoff before training and verifies strict bounds. See `docs/SCAN_CONTEXT_BENCHMARK.md` for the corrected observer policy.
