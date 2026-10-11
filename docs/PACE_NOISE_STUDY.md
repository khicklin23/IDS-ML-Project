# Timing pace and noise: matched follow-up

This follow-up tests the two remaining problems recorded in [temporal robustness](TEMPORAL_ROBUSTNESS.md): false flags on very slow harmless polling, and inconsistent results when timing is noisy.

## What is being tested

Three fresh seeds, 701, 733, and 769, generate independent datasets. Each dataset is used unchanged by three model preparations:

| Preparation | Timing inputs |
|---|---|
| Hybrid | Gap in seconds, plus gap relative to the session's average pace |
| Hybrid log | The same relative gap, plus a compressed absolute gap |
| Relative only | Relative gap; absolute pace is removed |

Compression uses `log(1 + gap)`. It turns a tenfold increase in seconds into a much smaller numerical increase. It does not decide whether traffic is harmless. The network still learns that decision from examples. Relative gaps are calculated from the original seconds before any compression.

Every candidate uses the existing six labels and the same small 32-unit session GRU. Training allows 200 epochs, waits at least 60 before early stopping, uses patience 20 and gentler class weighting (`class_weight_power=0.5`). The saved checkpoint has the lowest validation loss, never the highest test score.

Training contains ordinary and stealth telemetry, 1,200 matched timing pairs and 300 harmless polling sessions. Validation contains 400 pairs and 100 polling sessions. Timing noise is split equally between four levels, 0.08, 0.16, 0.25 and 0.35. Base timing paces span 5–120 seconds; harmless polling spans 5–25, 30–120 and 180–600 seconds.

Tests include familiar/slower timing, new cycle lengths, subtle differences, much slower paired cycles, harmless polling at 650–1,000 and 1,500–3,000 seconds, and noise beyond the training range. Paired sessions retain exactly the same individual feature measurements and endpoints; only interior order differs.

## Comparisons and checks

- Beta 2 is retrained on the same data and inputs as each Sona variant. It still predicts individual flows before voting on the session.
- A simple timing-summary classifier uses predefined gap statistics. It is trained on the binary timing subset only; its results do not claim six-class performance.
- Sequence shuffling checks whether the neural advantage depends on order.
- Every saved model/preprocessor is reloaded and checked against saved predictions. Raw splits must match exactly between variants and remain disjoint between training and testing.
- Alert thresholds are chosen on validation alone. Native class predictions and threshold-based alerts are reported separately.

The same raw data is processed by several models. Flow totals in the final summary count unique generated data once per seed, not once per model.

## Predeclared eligibility

A candidate may be considered for the synthetic showcase only if every seed passes ordinary/stealth macro F1 of at least 95%, familiar timing averages at least 95% with no seed below 90%, heavier-noise timing averages at least 80% with no seed below 70%, and harmless polling stays at or below 1% false flags in every tested pace range. Subtle and much-slower timing must each average at least 70%. The strongest noise test is exploratory and must be disclosed.

If several preparations qualify, the predeclared preference is compressed hybrid, relative only, then ordinary hybrid. Seed 701 is preselected for a possible public artifact; no best-test-seed selection. Passing these criteria does not establish production IDS accuracy. A consistent public export would still be needed.

## Reproduce

Use a fresh output directory for each seed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.range_study \
  --out-dir data/generated/sona_pace_noise_reproduction/seed_701 --seed 701

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.range_study \
  --out-dir data/generated/sona_pace_noise_reproduction/seed_701 --seed 701 --only-relative
```

Repeat for seeds 733 and 769, then run `python -m ids_ml.experiments.summarize_range data/generated/sona_pace_noise_reproduction` in the project environment.

Existing public model scores remain on Overview. Model Progression presents verified experimental summaries separately; its numbers do not silently relabel the public model.

The eligibility gate uses native class guesses. A separate alert threshold, chosen to limit false alerts on validation traffic, can miss more attacks. It is not interchangeable with classification accuracy; both outcomes are retained in the CSV. A candidate passing the gate does not certify a production alert policy.

## Completed relative-pace branch

Across all three fresh seeds, relative-pace candidates correctly classify 99.00% of familiar timing sessions against Beta 2’s 50%. Heavier-noise classification accuracy is 96.64%, and none of 1,800 harmless polling sessions beyond the training pace range receives an attack class. All three primary reload/data audits pass. These scores refer to separately trained candidates, not the existing Overview model.

Small timing differences remain weaker: 70.28% mean accuracy, with individual runs from 62.67% to 80%. Noise beyond the new training range averages 76.42%. A simple classifier using predefined timing summaries remains a serious comparison: it can perform better on subtle patterns. The evidence supports an advantage over individual-flow Beta 2 on these order-dependent cases, not universal neural-network superiority.

Shuffling the relative-pace candidates’ familiar timing sequences reduces mean accuracy to 50.75%. This control preserves each session’s measurements while disrupting their order. It supports the conclusion that the network is using sequence order.

With thresholds chosen to target at most 1% false alerts on validation, heavier-noise attack recall ranges from 88.67% to 96.17%; extreme-noise recall falls to 31.5%–53.83%. Both unseen harmless polling ranges still have zero threshold-based false alerts. These alert outcomes are lower than the classification accuracies and should not be substituted for them.

## Final selection and saved results

The complete study contains **2,171,844 unique flows and 44,048 sessions**, counted once per seed. All nine primary artifact audits and all raw-split comparisons passed. **Relative only** is the sole preparation to pass every predeclared eligibility criterion. Compressed hybrid still incorrectly labels 15/900 harmless 650–1,000-second sessions and 66/900 harmless 1,500–3,000-second sessions; its subtle-pattern mean is 67.33%. Ordinary hybrid performs worse on the slow harmless cases.

The relative candidate’s subtle-pattern result only narrowly clears the 70% mean target, and its weakest run scores 62.67%. Passing this synthetic gate is a reason to consider a consistent public export, not a claim that every remaining problem is solved. Seed **701** remains the prospectively chosen artifact seed; no best-seed replacement. Its weights and preprocessor are saved under `data/generated/sona_pace_noise_v1/seed_701/relative/`.

Full tables: [RESULTS.md](../data/generated/sona_pace_noise_v1/RESULTS.md). Every seed, shuffled control and validation-selected alert outcome: [per_seed.csv](../data/generated/sona_pace_noise_v1/per_seed.csv). Audit records, environment versions and source hashes: [summary.json](../data/generated/sona_pace_noise_v1/summary.json). Generated artifacts remain local and ignored by Git.

Model Progression now bundles a small verified summary in `ui/research.json`, separate from the public Overview report. Next backend step is a coherent full-size showcase export using the chosen preparation, followed by report/raw/comparison regeneration together; do not copy experimental scores into existing public artifacts. Before production-oriented claims, test more subtle patterns, tougher noise, additional independent data, and partial-session behavior.
