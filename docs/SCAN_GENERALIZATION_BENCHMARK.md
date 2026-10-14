# Traffic duration and background generalization

## Predeclared protocol

Fresh seeds: 1103, 1129, 1151. Per seed: 3,200 training windows, 3,200 validation windows, and 1,600 windows for each of five tests. Targets are balanced within each generated condition. Training and validation contain equal numbers from four independently generated cells: 30 or 60 minutes, each with lower or higher background counts. The lower count range is 48 to 84 connections and the higher range 64 to 116, plus 8 to 28 probe-like connections for either target.

Tests use fresh identities and independent random samples: familiar 30-minute/lower-background traffic, duration-only 60-minute/lower-background traffic, background-only 30-minute/higher-background traffic, both changes, and both changes with more legitimate inventory. These counts represent traffic volume per window, not equal connections per minute. Inventory is 25% of benign behavior in the last test, compared with 5% otherwise.

The existing generator supplies telemetry; the observer cutoff is applied uniformly before rescaling timestamps. Changing duration scales event times and gaps while keeping connection measurements and counts unchanged for a fixed seed. Dense conditions start from the higher-count generator. This preserves the original conditional behavior generation, including its absolute burst/jitter settings, before rescaling; conditions are controlled simulator transformations rather than recorded traffic. Uniform capture bounds, disjoint identities, and chronological ordering are audited.

## Candidates

All candidates are retrained on the same mixed training environments:

- Raw GRU: original permitted inputs.
- Relative-timing GRU: remove the absolute prior gap; add current gap and elapsed time relative to the mean strictly earlier positive gap. Before a prior positive gap exists, both features are zero. Flow duration itself remains an input.
- Small-history GRU: relative timing plus failure fraction and distinct failed ports in the previous 32 observations, and whether the current port is new.
- Small-history neural pooling: identical features and window supervision, but order-independent aggregation. History is supplied by causal feature calculations.
- Summary boosting: the existing strong full-window summaries, 150 iterations, 15 leaves, L2 penalty 1.
- Temporal heuristic: original diverse-failures-over-time score.

GRU hidden size remains 32; maximum 80 epochs, minimum 20, patience 12, batch 128, learning rate 0.001. Pooling uses the same schedule. Sequence cap 160 rejects oversized windows. Train-only normalization/encoding, validation loss checkpoint selection, and independent validation-selected 1%, 2%, and 5% false-alarm thresholds. All test thresholds are frozen. Report actual test false-alarm rates and every seed, including failures. Validation contains 1,600 benign windows per seed, twice the previous study's benign calibration count.

No ordinary-only training control is included in this study. Comparing its results with earlier studies cannot isolate the causal effect of augmented training, since seeds and sample sizes differ. Comparisons within this study isolate representation/learner differences on matched training data. Tests cover combinations included in training, with fresh environments and samples; they do not establish robustness to every unseen duration or traffic volume.

## Artifacts and verification

Run with the existing virtual environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.mixed_scan_generalization \
  --out-dir data/generated/mixed_scan_generalization_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.audit_scan_generalization \
  data/generated/mixed_scan_generalization_v1
```

The new output directory contains frozen configuration/source hashes, raw partitions, model weights, train-only fitted preprocessors, boosting model, validation thresholds, histories, predictions, and per-seed results. The audit reconstructs saved models and reproduces scores, thresholds and metrics. Dedicated tests check causal prefixes, label independence, timing scale invariance, independent condition transformations, balanced training cells, capture bounds, and complete artifact replay.

This is a separate binary window-classification experiment. Public Sona is still the saved six-category GRU. Its checkpoint, dashboard report, UI, and HTML release are not replaced. Legitimate inventory and scanning can remain indistinguishable from these measurements alone. Better detection must be assessed alongside actual false alarms, not only recall. No final outcome is assumed before running the study.

## Completed training results

The study generated **4,202,045 flow records in 43,200 windows** across three seeds. Each test condition has 4,800 windows in total, half scan-containing and half benign. Training and validation together contain 19,200 windows. All candidates use identical data; no settings changed after the first seed's results.

At the strict 1% validation budget, equal-weight seed means are:

| Condition | Raw GRU recall / actual false alarms | Relative-timing GRU recall / actual false alarms | Summary boosting recall / actual false alarms |
| --- | ---: | ---: | ---: |
| Familiar | 28.04% / 0.96% | 49.46% / 1.38% | 25.58% / 1.00% |
| Longer duration only | 40.50% / 1.50% | 50.17% / 1.50% | 24.67% / 1.21% |
| Higher background count only | 27.25% / 1.00% | 52.33% / 1.79% | 27.17% / 0.96% |
| Both changes | 41.25% / 1.71% | 51.00% / 1.71% | 23.96% / 0.96% |
| Both changes, more inventory | 42.71% / 5.00% | 52.62% / 5.21% | 24.54% / 2.58% |

The relative-timing candidate improves average recall in every condition. The longer-only and combined conditions provide the clearest comparison: recall gains of 9.67 and 9.75 percentage points respectively at identical aggregate actual false-alarm rates. Familiar and higher-background gains also involve higher false alarms, so their gains cannot be described as free. Aggregate rate equality does not imply identical flagged records or equality in every seed.

The improvement is not universal per seed. Seed 1129 favors the raw GRU on duration-only (45.62% versus 41.75% recall) and combined (52.00% versus 43.38%) traffic. Across seeds, relative familiar recall ranges from 41.50% to 54.62%, and relative combined recall from 43.38% to 62.25%. Three seeds are useful confirmation, not a guarantee.

### Threshold tradeoffs

| Relative-timing candidate condition | 1% budget recall / actual false alarms | 2% budget recall / actual false alarms | 5% budget recall / actual false alarms |
| --- | ---: | ---: | ---: |
| Familiar | 49.46% / 1.38% | 59.67% / 2.21% | 80.46% / 5.96% |
| Longer duration only | 50.17% / 1.50% | 59.88% / 2.25% | 80.92% / 6.08% |
| Higher background count only | 52.33% / 1.79% | 62.50% / 3.04% | 82.29% / 5.67% |
| Both changes | 51.00% / 1.71% | 62.12% / 2.25% | 81.96% / 5.42% |
| More inventory | 52.62% / 5.21% | 61.50% / 7.33% | 82.38% / 13.08% |

Raising the budget changes the decision threshold, not the trained weights. Higher recall at the loose setting comes with substantially more false alarms.

### What failed and what remains

The small-history GRU did not improve the raw control: strict familiar recall is 20.38% at 1.17% false alarms, and combined recall is 17.04% at 0.96%. Small-history pooling reaches 17.42% at 1.25% familiar and 16.50% at 2.12% combined. Adding these history fields is not recommended from these results. Full tables retain these candidates and the temporal heuristic rather than hiding them.

All GRUs retain hidden size 32. Parameter counts are raw 5,250, relative 5,346, and small-history 5,634; pooling has 1,362. The strongest candidate adds only 96 input-projection parameters, not a wider memory layer. This is an input/training experiment, not evidence that a much larger network is necessary.

Inventory remains ambiguous. The relative candidate flags 115 of 607 legitimate inventory windows in the inventory-heavy test, **18.95% of that subtype**, despite only 5.21% false alarms overall. At the strict setting, it also misses roughly half the scan windows. Neither number supports a production-ready detector or replacement of the public six-category model.

These results address the earlier collapse within the tested training conditions: relative timing maintains about 50% mean recall across the four ordinary test cells. They do not isolate why it improved compared with the older study, because training augmentation, timing representation, seeds and sample sizes changed. The matched raw control supports a within-study representation benefit. No test here demonstrates behavior at completely unseen durations or volumes.

## Next experiment

Confirm with fresh seeds, unseen intermediate/longer durations, and matched pooling on relative-only inputs. Include ordinary-only versus mixed-condition training controls with equal training counts and a shared validation/test protocol to isolate augmentation. Retain strong summary baselines and representative benign calibration, and report inventory errors separately. Do not widen the network or promote the candidate solely from this result. Integrating any confirmed benefit into public Sona will require six-category training and full regression evaluation, not copying binary scan scores into the dashboard.

## Verification

All 119 backend tests passed in 257.05 seconds. Dedicated tests include complete model reload and threshold replay. All three full independent audits passed: 108 prediction replays and 324 operating-point checks reproduced scores, validation-selected thresholds, and behavior-specific metrics. Capture bounds, disjoint identities, train-only preprocessing, saved source hashes, aggregate results and frozen implementation hashes were verified. Public checkpoint/report/raw-manifest/research/release hashes and earlier mixed-scan artifacts remain unchanged. Complete all-candidate tables are local at `data/generated/mixed_scan_generalization_v1/RESULTS.md`; `per_seed.csv` retains all 324 candidate/partition/budget rows and `validation_cells.csv` retains 216 calibration-cell rows. The latter distinguishes the mixture-level threshold budget from each individual cell's actual false alarms. No installs, commits, pushes, checkpoint promotion, UI refresh, or HTML release build occurred.
