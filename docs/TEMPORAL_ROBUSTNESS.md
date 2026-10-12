# Temporal robustness: fixes and remaining limits

The backend now supports a minimum learning window and saved session-relative timing. These changes address the early-stopping and slower-traffic failures found in the first temporal benchmark. Existing public model files, dashboard data, and release HTML are preserved.

## What changed

- **Training gets time to learn.** `SequenceTrainConfig.min_epochs` prevents an early plateau from ending training immediately. The saved model still comes from the lowest validation loss, including checkpoints before that minimum. Default report/comparison entry points now wait at least 45 epochs; explicitly supplied configurations retain control.
- **Timing can be relative to the session's pace.** A gap is divided by the mean positive gap within that completed session. The encoder is fitted on training data only and saved with this transformation, so inference uses the same preparation.
- **Absolute timing can be retained.** `timing_mode='hybrid'` adds a derived relative-gap feature alongside the original gap in seconds. `relative` replaces the gap; `absolute` preserves the historical feature preparation. Absolute remains the default; timing modes are opt-in until broader validation supports promotion.
- **Rare classes can receive gentler emphasis.** `class_weight_power=1.0` keeps the existing inverse-frequency weighting. `0.5` uses its square root: rare families still receive extra emphasis, but do not outweigh difficult common families as strongly. This changes training loss, not alert severity.

Relative and hybrid modes use **complete closed sessions**. They are appropriate for the current precomputed reports, not a claim of live, future-free detection. Session IDs group observations; they never become input features. Labels, IPs, timestamps, and generator provenance remain excluded.

## Three fresh-seed robustness runs

Seeds 211, 239, and 271 generated **1,639,604 flows**, with 23,400 held-out sessions. Both timing versions received the same broader training data: more cycle widths and contrasts, plus scheduled benign polling. That makes their comparison a test of normalization on a shared recipe, not a comparison with the earlier, easier dataset.

Accuracy is the percentage of sessions classified correctly. Each timing test contains paired benign and attack-like sessions with identical individual flow measurements and endpoints; order is the distinguishing information.

| Held-out timing test | Beta 2 | Sona raw timing | Sona relative timing | Timing-summary classifier |
|---|---:|---:|---:|---:|
| Familiar patterns | 50.00% | 95.03% | 98.08% | 98.19% |
| Slower pace only | 50.00% | 52.75% | **98.03%** | 98.42% |
| More timing noise | 50.00% | 81.19% | 77.89% | **92.36%** |
| Unseen cycle width | 50.00% | 64.00% | 49.78% | 48.17% |
| Very subtle short/long differences | 50.00% | 55.61% | 58.64% | **81.81%** |
| Combined slower/noisier pattern | 50.00% | 50.56% | **84.42%** | 92.00% |

Relative Sona produced **zero false alerts in 1,800 synthetic benign polling sessions**, including slower polling. Shuffling the tested sequences removed the classification advantage. The full saved-data audit passed: paired feature collections/endpoints match before and after normalization, timestamps agree with gaps, and splits are disjoint.

The pace improvement held across all three seeds (96.50–98.92% accuracy). Noise performance varied substantially (67.17–85.25%). Normalization is a useful pace fix, not a universal robustness fix, and the timing-summary baseline remains stronger on several cases.

[Full study results](../data/generated/temporal_robustness_v1/RESULTS.md) · [machine-readable report](../data/generated/temporal_robustness_v1/report.json)

## Six-class integration checks

The first relative-only candidate (seed 313) missed one normal outbound-transfer session: macro F1 96.89%, versus Beta 2's 100%. Both scored 100% on its stealth test. That failure is preserved.

A larger fresh hybrid candidate (seed 347; **53,328 flows**, 114 held-out sessions) scored 100% macro F1 on normal and stealth traffic, matching Beta 2. This checks six-class compatibility; it does not establish hybrid's superiority, particularly because these candidates use different seeds and dataset sizes.

A separate joint candidate (seed 401; **523,779 flows**) mixed standard and stealth examples with paired timing examples under the existing six labels. Both architectures received the same data and hybrid inputs. The first joint training recipe was inadequate: Sona scored 52.00% on familiar timing and 53.83% on slower timing. Its standard-traffic accuracy was 97.67%, with one outbound-transfer session confused with service flood; stealth accuracy was 100%. It also falsely alerted on 11 of 300 slower benign polling sessions. This candidate was not promoted.

Validation loss was still improving near the end of that run. Three fresh follow-ups (427, 451, 479) tested 160 epochs and gentler weighting. They learned familiar timing well (98.58–99.25%) and all matched Beta 2's 100% standard/stealth accuracy. However, they falsely flagged 58.00–66.33% of slower harmless polling sessions. None was promoted. These runs change seed, epoch window, and weighting together, so they compare recipes rather than isolate weighting's effect.

A threshold selected for 1% false alerts on seed 427's validation data still falsely flagged 44.67% of slower test polling and missed 47.50% of familiar timing attacks. Raising the threshold did not repair the missing training coverage.

The next recipe expands paired training paces from 5–25 to 5–120 seconds, and benign polling to three ranges: 5–25, 30–120, and 180–600 seconds. Fresh seeds 503, 541, and 577 were declared before their test outcomes. They retain the same classes, network, weighting, and 160-epoch limit. An extra paired test uses 150–300 second base gaps, beyond the paired training range. Original failures remain saved unchanged.

### Expanded-coverage results

The three revised six-class candidates generated **1,741,958 flows**, including **22,762 held-out sessions**. Both architectures receive the same training examples and hybrid inputs. These are newly trained candidate weights, not scores for the unchanged public dashboard snapshot.

| Held-out test | Beta 2 accuracy | Sona accuracy |
|---|---:|---:|
| Standard traffic | 100.00% | 100.00% |
| Less obvious attacks | 93.07% | 100.00% |
| Repeated timing | 50.00% | **98.97%** |
| Slower, noisier cycles | 50.00% | **85.53%** |
| Long cycles included in training | 50.00% | 98.97% |
| Longer cycles absent from training | 50.00% | **94.72%** |
| Small timing differences | 50.00% | 69.97% |
| Much slower cycles: new paired pace | 50.00% | 77.61% |
| Harmless slow polling: correctly ignored | 66.67% | **100.00%** |

Sona falsely flagged **0 of 900** harmless slow polling sessions, down from 560 of 900 in the preceding recipe. This is an observed synthetic test result, not a guaranteed false-alert rate. Standard and less-obvious tests cover all six classes; Sona's mean classification macro F1 is 100% on both, versus Beta 2's 100% and 92.58%. Those ordinary test groups are small (262 sessions combined); the timing groups provide the larger controlled comparison.

The clearest recap example is the paired timing test: every benign/attack pair has the same individual flow measurements, so Beta 2's flow-by-flow majority vote cannot separate them. Sona learns from their order. This establishes an advantage over Beta 2's current input processing; a classifier using hand-built timing summaries can also recognize order, as the separate robustness study shows.

Subtle timing and very slow cycles remain weaker, and results vary between seeds. Broad training coverage fixed the slow-polling failure within the tested range; it did not establish general intrusion detection strength on real networks.

### Frozen-model stress checks

Fresh supplemental checks added 216,388 flows and 4,500 sessions without retraining. Heavier timing noise reduced Sona's mean accuracy to **74.19%**, with a wide 62.75–89.67% range; Beta 2 remained at 50%. Harmless polling at **650–1,000 seconds**, beyond the training polling range, exposed another failure: seed 541 falsely classified **185 of 300** sessions (61.67%). Seeds 503 and 577 correctly ignored all 300 each. Across all three runs, that is **185 false flags in 900** sessions (20.56%), versus zero for Beta 2 on this particular control.

Alert thresholds chosen exclusively from validation preserved familiar-timing attack recall at **99.72%** on average, with **0.72%** false alerts among paired benign timing sessions. But they did not repair the unseen polling problem: seed 541 still generated 161 of 300 false alerts. Its harder-noise attack recall fell to 24.83%, illustrating why high classification accuracy and cautious alert thresholds must be reported separately.

Shuffling familiar timing removes the advantage (mean Sona accuracy **49.83%**). All three candidates passed split/schema/paired-feature checks and nine saved-model inference reload checks each. Supplemental paired inputs were also audited before inference. The complete seven-candidate joint record covers **4,047,000 flows**; the separate raw-versus-relative robustness study adds 1,639,604 more.

**No candidate has been promoted to the public dashboard.** The next training experiment should reduce sensitivity to absolute gap size while retaining relative timing, and add stronger noise coverage. A logarithmic gap feature is a hypothesis to test, not an established fix. Success needs repeatable results across fresh seeds, harmless out-of-range polling, and difficult timing cases, not just the highest score from one seed.

An exploratory pace intervention supports that direction: for seed 541, the same 300 harmless polling sessions went from 115 correct to 300 correct when gaps changed from 650–1,000 to 65–100 seconds. All encoded inputs except the absolute gap stayed equal within float32 tolerance. The original 185 errors were all beaconing predictions. This isolates a pace sensitivity; it is not independent new evidence or a deployed correction.

[Complete results and seed ranges](../data/generated/temporal_fix_summary_v2/RESULTS.md) · [per-seed CSV](../data/generated/temporal_fix_summary_v2/per_seed.csv) · [configuration, thresholds, and audits](../data/generated/temporal_fix_summary_v2/summary.json)

## Use the backend options

```python
from ids_ml.features import build_preprocessor
from ids_ml.models.sequence import SequenceTrainConfig

# Fit this on complete training sessions only; save it with the trained model.
preprocessor = build_preprocessor(train_df, timing_mode="hybrid")
config = SequenceTrainConfig(
    seed=427, max_epochs=160, min_epochs=60,
    patience=20, class_weight_power=0.5,
)
```

`run_comparison(..., sona_timing_mode='hybrid')` and `build_session_report(..., timing_mode='hybrid')` support the same persisted preprocessing. New Sona metadata records timing mode and encoded input names/width; comparison exports save `v4_preprocessor.pkl` separately. Existing absolute-timing artifacts continue to load.

Those entry points still train their ordinary/stealth recipe; enabling hybrid alone does not add the temporal examples. The full joint candidate recipe is in `ids_ml.experiments.joint_temporal`, with `--pace-augmentation` for the expanded coverage.

Recorded experiments live in separate gitignored directories. Use fresh directory names for reruns:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.temporal_robustness \
  --out-dir data/generated/temporal_robustness_reproduction

.venv/bin/python -m ids_ml.experiments.audit_robustness \
  data/generated/temporal_robustness_reproduction

# Six-class candidate with the wider benign/attack pace coverage.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.joint_temporal \
  --out-dir data/generated/sona_joint_reproduction --seed 503 \
  --epochs 160 --class-weight-power 0.5 --pace-augmentation

.venv/bin/python -m ids_ml.experiments.audit_joint \
  data/generated/sona_joint_reproduction

# Frozen-model, independently generated noise/very-slow benign checks.
.venv/bin/python -m ids_ml.experiments.extra_stress_joint \
  data/generated/sona_joint_reproduction

# Optional: measure alerts at a validation-selected 1% false-alert target.
.venv/bin/python -m ids_ml.experiments.operating_point_joint \
  data/generated/sona_joint_reproduction
```

The dashboard has not been switched to any candidate. A future refresh must describe the chosen timing mode accurately and use the matching saved preprocessing. Synthetic order recognition is evidence of a model capability, not proof of malicious intent or production detection accuracy.
