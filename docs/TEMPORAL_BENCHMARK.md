# Where Sona's memory helps

This experiment is separate from the public showcase. It compares freshly trained versions of the **Beta 2 architecture** (a network that classifies individual flows) and the **Sona architecture** (a GRU that reads an ordered session). It does not evaluate or replace the deployed six-class model weights.

A flow is one network conversation. A session is a series of conversations. The question here is whether their **order** can provide information that individual flow measurements do not.

## Recorded default-training results

The five-seed experiment generated **1,733,002 flows**, including 20,000 unseen test sessions across its familiar and shifted timing splits.

| Unseen data | Beta 2 accuracy | Sona accuracy | Simple timing-summary accuracy |
|---|---:|---:|---:|
| Familiar timing ranges | 50.00% | 89.50% | 100.00% |
| Slower, noisier timing; new repetition width | 50.00% | 49.95% | 80.03% |

Four Sona runs scored 98.80–99.65% accuracy on familiar timing. One scored 50.25%: early stopping selected epoch 2, before learning emerged. Shuffling Sona's unseen flow order brought average accuracy down to 49.87%; training and testing it with shuffled order gave 50.43%. The last-flow classifier scored 50%. Those controls support an advantage from reading the sequence, rather than from individual measurements or endpoints.

**The current training settings are inconsistent, and Sona did not handle the shifted timing reliably.** The simple timing-summary classifier also outperformed it. These findings support further temporal-model development, not a claim that Sona is superior to every simpler method.

Detailed results: [default-training report](../data/generated/temporal_patterns_large_v1/RESULTS.md). Raw scores and training histories remain available beside that report.

## Fresh-seed longer-training follow-up

Validation histories showed that successful runs needed more than 20 epochs before learning emerged. A separate follow-up allowed 60 epochs without an early-stop cutoff, still selecting the saved checkpoint by validation loss. It used fresh generation/training seeds **113, 137, and 163**, the same split sizes and controls, and another **1,036,076 flows**. The attack definition and shifted test were unchanged. CPU library threads were capped at two; exact commands and package versions are recorded with the output.

| Unseen data / control | Beta 2 accuracy | Sona accuracy | Simple timing-summary accuracy |
|---|---:|---:|---:|
| Familiar timing ranges | 50.00% | **99.02%** | 100.00% |
| Familiar timing, Sona's test order shuffled | 50.00% | 50.15% |, |
| Slower, noisier timing; new repetition width | 50.00% | 50.17% | 80.13% |

All three Sona runs scored 98.85–99.20% on familiar timing. Its average macro F1 was 99.02%. At thresholds selected to allow at most 1% false alarms on validation, it detected 99.20% of unseen attack-like sessions, with an actual benign false-alarm rate of 1.40%. Beta 2 detected 1.37% while generating the same 1.37% false-alarm rate: its scores could not distinguish paired classes.

The longer-window runs show that this architecture can consistently learn the ordered pattern in these three fresh datasets. They do not establish that longer training alone will solve reliability across every seed or dataset. The independent five-seed default results remain part of the record. Longer training also did **not** resolve the unfamiliar-timing failure.

Across both large experiments: **2,769,078 flows**, 57,600 sessions, and 32,000 held-out test sessions. This count excludes the smaller pilot. Each experiment includes independent train/validation/test data, but the benchmark design was developed first; these results are not an external blind security evaluation.

Detailed results: [longer-training report](../data/generated/temporal_patterns_long_training_v1/RESULTS.md). For a project recap, a defensible statement is:

> On a controlled synthetic timing test, the Sona architecture learned differences in flow order with 99.02% accuracy across three fresh runs. The per-flow Beta 2 architecture scored 50%. Shuffling the order removed Sona's advantage. Unfamiliar timing and stronger simple baselines remain areas for improvement.

## A comparison without an obvious shortcut

Each repeating, beacon-like session has a benign shuffled partner with exactly the same measurements, number of flows, and first and last flows. The timing pattern repeats short and long gaps, with varied pace, starting position, block width, and noise. Packet counts, byte counts, ports, and durations also vary, but cannot reveal the label: both members of a pair contain the same flow feature collection.

This is a controlled capability test. Calling repeating traffic attack-like does not establish malicious intent. Real scheduled software can also produce repeating traffic.

The larger experiment uses five independent generation and training seeds: 11, 29, 47, 71, and 101. Each seed has:

| Split | Sessions | Purpose |
|---|---:|---|
| Training | 2,400 | Learn model weights and preprocessing |
| Validation | 800 | Select training checkpoint and alert threshold |
| Unseen familiar timing | 2,000 | Measure performance without reusing training sessions |
| Unseen shifted timing | 2,000 | Test slower traffic, more timing noise, and a new repetition width |

Sessions contain 32–64 flows. Paired sessions stay together within one split. Labels, identifiers, and timestamps never enter the feature arrays. Preprocessing is fitted on training data only. Both networks use the existing training implementations and receive the same measurements and split membership. Beta 2's session prediction uses majority voting over its individual flow predictions.

## Controls that make the result useful

- **Shuffle Sona's test order:** keep measurements and endpoints, remove the learned sequence.
- **Train and test Sona with shuffled order:** remove sequence information throughout the experiment.
- **Last-flow classifier:** check whether the final conversation alone provides a shortcut.
- **Timing-summary classifier:** a simple logistic classifier using gap statistics, repeated-timing calculations, and adjacent-gap changes. This tests whether explicitly calculating timing features also solves the problem.

Accuracy is the percentage of sessions classified correctly. Macro F1 balances mistakes for each class, counting both missed attacks and false alarms. AUC measures whether attack-like sessions receive higher scores than benign sessions; 50% means chance ranking. Different threshold choices can change accuracy and F1 without changing the underlying ranking.

A second comparison chooses each model's threshold on validation data, allowing at most 1% benign false alarms there. Reports show the **actual** false-alarm rate on unseen data beside attack recall. The unseen rate can exceed 1%; the validation target is not a guarantee.

## Reproduce and inspect

Run from the repository root with the existing virtual environment. Use a new output directory: the experiment refuses to overwrite previous runs.

```bash
.venv/bin/python -m ids_ml.experiments.temporal_patterns \
  --out-dir data/generated/temporal_patterns_large_v1

.venv/bin/python -m ids_ml.experiments.summarize_temporal \
  data/generated/temporal_patterns_large_v1

.venv/bin/python -m ids_ml.experiments.audit_temporal \
  data/generated/temporal_patterns_large_v1
```

The recorded follow-up command was:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.temporal_patterns \
  --out-dir data/generated/temporal_patterns_long_training_v1 \
  --seeds 113 137 163 --epochs 60 --patience 60
```

Both recorded output directories already exist. Choose a fresh directory for reruns, then pass it to the summarizer and audit commands.

The local output contains:

- `RESULTS.md`: readable comparison, controls, and individual seed results.
- `report.json`: configuration, metrics, training histories, and aggregate statistics.
- `per_seed_metrics.csv`: spreadsheet-ready measurements.
- `seed-*/`: raw split Parquet files, trained weights, preprocessing, timing-summary classifier, prediction scores, and `results.json`.
- `checkpoint.json`: completed runs, available before the full experiment finishes.
- `environment.json`: package versions and the command used for this recorded run.
- `audit.json`: full saved-data audit of split separation, matched feature collections/endpoints, coherent timestamps, and paired prediction equality.

These generated data and model artifacts are excluded from source control. The small pilot remains separately available at `data/generated/temporal_pilot`; it failed to train Sona reliably and is not discarded from the record.

The pilot predates a numerical correction: averaging Beta 2 scores in float32 introduced tiny order-dependent rounding differences (up to roughly 0.00000012). Both large experiments use float64 accumulation so matched partners receive exactly equal scores. The strict prediction-pair audit intentionally rejects the old pilot artifacts; use the large experiments for quantitative comparisons.

## Next experiments

1. Make training more dependable: test a minimum learning window before early stopping, while continuing to select checkpoints on validation data. Evaluate with additional fresh seeds.
2. Test session-relative timing: measure each gap relative to the session's usual pace, while keeping absolute timing available where useful. Fit any learned transformations using training data only.
3. Separate the stress factors. Change only pace, only jitter, and only repetition width in different datasets to identify what causes failures.
4. Add real benign lookalikes: scheduled polling, keep-alives, backups, and telemetry. Add attacker jitter, missing observations, and mixed activity. Periodicity alone should not imply an attack.
5. Compare with timing-feature baselines throughout. A neural network advantage over a per-flow model does not establish an advantage over every simpler approach.
6. Integrate successful changes into the six-class training pipeline, then evaluate the actual deployed Beta 2 and Sona checkpoints on a fresh, shared benchmark. Include ordinary traffic so temporal improvements do not silently harm other detections.

Do not replace the public comparison with this experiment's numbers: its two-class task and deliberately paired data answer a narrower question.
