# Why session memory was added

Beta 2 learns from one connection at a time. More varied training helps it recognize slowed-down attacks, but voting on individual connections loses their order. Two sessions containing the same measurements in different orders therefore receive the same vote. A session-memory network can use that missing information. This was the reason for testing Beta 3 and then combining memory with broader training in Sona. It was not a speed improvement.

## Four-model follow-up protocol

This extends the existing audited pace/noise study, using its exact saved train, validation and test sessions for seeds 701, 733 and 769. It is an extension on known test cases, not a fresh independent confirmation. Earlier model snapshots are preserved.

| Candidate | Model view | Ordinary attack training | Timing practice |
|---|---|---|---|
| Beta 1 | Individual connections | Standard attacks | Shared timing pairs and harmless polling |
| Beta 2 | Individual connections | Standard and slowed-down attacks | Same timing pairs and harmless polling |
| Beta 3 | Connections in order, with memory | Standard attacks | Same timing pairs and harmless polling |
| Sona | Connections in order, with memory | Standard and slowed-down attacks | Same timing pairs and harmless polling |

Beta 2 and Sona reuse the existing saved relative-pace candidates. Beta 1 and Beta 3 are newly trained counterparts. Their training/validation frames remove only the slowed-down ordinary generation run; all timing examples remain identical. Their shared preprocessing is fitted only on this narrower training frame. All four use relative timing. This is a controlled retraining comparison, not a retest of the original public beta weights.

The Beta 1 training settings match Beta 2: up to 80 epochs, patience 10, batch size 512, 32-unit hidden layer, inverse-frequency class weighting. Beta 3 matches Sona: up to 200 epochs, minimum 60, patience 20, batch size 128, 32-unit GRU, class-weight power 0.5. Checkpoints use validation loss only. No test-score seed choice or recipe changes after outcomes. Every seed and weaker test is reported.

All four are evaluated on the same ordinary, slowed-down, timing, noisy, subtle and harmless-polling test sessions. Individual-flow models vote on a session; memory models predict it directly. Conditional timing scores use accuracy, not six-class macro F1. Artifact reloads reproduce predictions; test IDs/labels align exactly; source data hashes, split disjointness and matched-pair votes are checked. Shuffling checks whether Beta 3 uses sequence order. No model speed benchmark is claimed.

The primary UI comparison will show ordinary, slowed-down and timing accuracy separately. If a summary score is displayed, it averages those three category means equally and explicitly says so. This is a descriptive synthetic-test score, not a predeclared model-selection gate or a production IDS rating. Sona is not assumed to win every test.

## Reproduce

For each seed 701, 733 and 769:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.four_model_timing \
  --parent data/generated/sona_pace_noise_v1/seed_701/relative \
  --out-dir data/generated/four_model_timing_reproduction/seed_701
```

Then run `.venv/bin/python -m ids_ml.experiments.summarize_four_models data/generated/four_model_timing_reproduction`.

The first evaluation attempt used session names without their generation run IDs and incorrectly reported overlap between independent ordinary and stealth test runs. The check now uses the same canonical run/session identities as the parent audit, with a regression test. Evaluation resumed from the saved weights without retraining. The original per-epoch history arrays had not yet been persisted; the report records that limitation. Subsequent reproductions persist histories before evaluation, and `--resume-evaluation` reuses saved checkpoints after an audit interruption.

## Verified outcome

All three runs passed the complete audit: 13 shared test splits per run, all four saved models reloaded against their predictions, matched labels/session IDs, unchanged source hashes, and order-invariant Beta 1 votes on paired timing cases. The new session-identity regression passes.

| Candidate | Ordinary accuracy | Slowed-down accuracy | Timing accuracy | Equal-weight three-test score |
|---|---:|---:|---:|---:|
| Beta 1 | 100.00% | 100.00% | 50.00% | 83.33% |
| Beta 2 | 100.00% | 100.00% | 50.00% | 83.33% |
| Beta 3 | 100.00% | 91.67% | 99.19% | 96.95% |
| Sona candidate | 100.00% | 99.24% | 99.00% | 99.41% |

These means cover exactly the same 131 ordinary, 132 slowed-down and 3,600 timing test sessions. The other ten shared test categories are included in the complete evidence.

**Why move beyond Beta 2?** Its individual-connection vote gets the timing pairs half right because each pair has identical measurements in different orders. Adding memory in Beta 3 raises timing accuracy to 99.19%. Adding broader ordinary-attack training in Sona raises slowed-down accuracy from Beta 3’s 91.67% to 99.24%, while retaining roughly 99% timing accuracy. This supports the decision to combine the two approaches. It is evidence about these benchmark cases, not a guarantee of every future attack or faster execution.

Beta 3 is slightly stronger than Sona on familiar timing and several longer-cycle cases. Sona has the strongest three-category average, but does not win every test. Beta 1 and Beta 2 tie in this follow-up: their shared timing practice and relative inputs change the training setting, so the earlier improvement from Beta 1 to Beta 2 is still documented by the original traffic-test comparison. Do not fabricate a gap where the new data shows none.

Full means and ranges: [RESULTS.md](../data/generated/four_model_timing_v1/RESULTS.md). Every seed and shuffle control: [per_seed.csv](../data/generated/four_model_timing_v1/per_seed.csv). Models, training settings, raw-source hashes and audit records: [summary.json](../data/generated/four_model_timing_v1/summary.json). The local generated artifacts are ignored by Git.

Model Progression now shows all four counterparts with these same-session scores and the quantified rationale. Extended stress results are expandable. The public Overview model/report/raw dataset remain unchanged; no inference-speed measurement or public-model promotion was performed.
