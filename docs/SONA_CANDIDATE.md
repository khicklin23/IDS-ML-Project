# Sona candidate: frozen confirmation protocol

Owner: `codex-sona-candidate-20261013`. User authorized local implementation,
generation, training and evaluation on 2026-10-13. Claude owns the separate
`sona_evaluation` package and evaluation document. Public artifacts and earlier
experiments remain preserved. Git-kb assignment failed for the uncommitted
confirmation task; this ownership is explicitly established by the user's split.

## Protocol frozen before new test results

Fresh training seeds: **1301, 1327, 1361**. Each has 3,200 ordinary-only training
windows and 3,200 mixed training windows, with equal target counts. Mixed training
has equal 30/60-minute and low/high background-count cells. These are volume
conditions, not equal connections per minute. Shared validation: 3,200 windows
from those four cells plus an equal fifth inventory-heavy cell. This changes the
benign calibration mix deliberately before testing. Every training regime fits
its own preprocessor on its own training rows only.

Seven tests, 1,200 windows each per seed: fresh familiar, 60-minute, higher count,
combined, inventory-heavy, previously unseen 45-minute and 90-minute low-count
windows. Timestamp scaling uses the frozen simulator's capture-bounded conditions.
The inventory-heavy distribution includes 25% inventory among benign windows.
These remain synthetic intent labels with intrinsic overlap, not authorization
metadata inferred from labels.

Predeclared candidates: raw mixed GRU32; relative ordinary/mixed GRU32; relative
ordinary/mixed pooled MLP; relative mixed hierarchical memory; ordinary/mixed
summary boosting. The hierarchical candidate projects each event to 16 units,
summarizes blocks of eight with learned-feature mean/max, uses a GRU24 over
blocks, and combines its final state with attention over earlier block states.
This reduces recurrent steps, retains long-window evidence, and loses order
within each block. It is justified by sparse attacks mixed with distractors,
not by a claim that additional layers automatically confer reasoning.

Neural schedule: Adam 0.001, gradient norm cap 5, maximum 60 epochs, minimum 20, patience 12, batch
128, CPU two threads, deterministic seed. Validation loss chooses checkpoints.
Summary boosting: 150 iterations, 15 leaves, L2=1. Relative inputs remove absolute
prior gap, use strictly earlier positive gaps for scale, and retain elapsed
relative units. Even relative pooling has timing history in its inputs; it is
not a completely history-free control. Standard GRU32 is the small architecture
control; exact parameter counts and efficiency are reported, not assumed equal.

Keep validation-selected 1%, 2%, and 5% false-alarm thresholds. Report all test
recalls, actual false alarms, subtype errors and per-seed ranges. Finalist
selection uses mean validation recall at the strict budget only; equal scores
prefer fewer parameters. Test outcomes cannot change selection. No public
promotion without independent evaluation and separate six-category regression.

Binary success gate: relative mixed GRU improves strict-budget recall over both
relative ordinary GRU and matched mixed pooling by at least five percentage
points on at least two of familiar/45-minute/90-minute tests, in at least two
seeds; actual false alarms must be within 0.5 percentage points of controls and
at most 2% on those tests. Inventory-heavy actual FPR must be at most 2%, and
inventory-subtype FPR at most 10%. The hierarchical model must meet those safety
limits and improve recall or reduce model inference time by at least 20% without
losing more than two recall points versus GRU32. These are research gates, not
production accuracy standards. Failed gates are retained; no test-tuned rescue.

## Inference contract for independent evaluation

`from ids_ml.experiments.sona_candidate.inference import CandidatePredictor`

`predictor = CandidatePredictor.load(path_to_candidate_directory)`

`predictor.predict(frame)` returns a dictionary with `session_ids` (strings),
`probabilities` (N by 2, benign then suspicious), and `scores` (suspicious column).
The frame uses existing permitted telemetry fields and `session_id` to group
chronological observations. Labels, behavior and observation metadata are not
required for prediction. Empty input, nonchronological sessions and over-cap
sequences are rejected. `predictor.manifest` contains architecture, cap, feature
names, thresholds and version. Artifacts are experimental binary-window models,
incompatible with the public six-category report. Only load trusted local pickle
preprocessors; checkpoints use Torch weights-only loading.

Each candidate directory contains `manifest.json`, `model.pt` or `model.pkl`,
`preprocessor.pkl` where applicable, and `history.json`. Completed study writes
`finalists.json` listing validation-selected candidate directories. This document
will append commands, audits, efficiency results and final recommendation.

## Progress

**Pre-test amendment, 2026-10-13:** before any new test outcomes are inspected,
add a parameter-matched pooling control (hidden widths96/32) for both ordinary
and mixed regimes. It uses identical relative inputs, training counts, optimizer,
gradient cap and checkpoint schedule. For the expected21inputs this has5282
parameters versus GRU32's5346 (1.2% difference). This addresses the capacity
confound in the smaller32/16pool. It is an additional confirmation control,
not eligible for the already frozen finalist selection. Train it separately
without modifying the in-flight eight-candidate source or artifacts. Preserve
all results and audit its saved predictions and thresholds independently.

**Second bounded amendment, before inspecting test outcomes:** seed1301
validation selects relative mixed GRU (29.375% strict-budget recall) over the
block hierarchy (14%). This motivates one additional memory design that avoids
discarding event order in blocks: a32-unit causal dilated event encoder and two
shared learned retrieval hops over its event states. No claim of human reasoning
is implied. Compare it with GRU40 as a parameter-matched control (about7.5k
parameters each), using identical mixed3200training windows, validation3200,
optimizer/schedule and all frozen test partitions. These are the final additional
architecture arms; no further architecture search after seeing tests.

Stage2 selects among the original neural candidates and these two arms using
only three-seed mean strict-budget validation recall, breaking ties by fewer
parameters. Original stage1 selection/results remain preserved. The final
independent adapter will use that frozen stage2 choice, never a test winner.
The retrieval candidate's quality gate requires at least5points recall gain
over GRU32 on at least two familiar/45/90-minute cells in two seeds, FPR at most
2% and within0.5points of GRU32, plus inventory overall FPR at most2% and subtype
FPR at most10%. Efficiency remains independently measured. A failed gate prevents
six-category promotion work even if an architecture wins validation selection.

**Coordination status: independent evaluation COMPLETE; promotion rejected.** The final frozen
choice is **GRU40 with relative inputs**, selected on three-seed validation only.
The primary agent completed a separately reported sealed replication at
`data/generated/sona_candidate_independent_final_v1` using Claude's unchanged
evaluation suite. Claude's evaluator-owned run is the primary assessment. Candidate source and this document
remained unchanged during both sealed runs. Public promotion remains rejected by the
binary quality gate; this independent run assesses architecture capabilities.

The processor computes only the requested relative timing features, avoiding
unused historical counters. Exact equivalence with the previous frozen timing
calculation, causal prefixes, scale invariance and label blindness are tested.
The first artifact smoke test passed all 64 prediction and 192 operating-point
replays, but its resume test correctly rejected source edits made while the
test was running. Verification is being rerun after source stabilization.
Final architecture selection uses mean validation recall across seeds; individual
seed selections are retained as diagnostics only.

All123backend tests passed301.78s. The updated independent-suite adapter's own
round-trip test passed64.00s. Full training is in progress; no test outcomes have
been used to revise the frozen protocol. A separate inference-only optimization
avoids repeated table slicing/validation; packaging will require exact feature
arrays on every evaluated partition and exact predictions after reload. This
optimization changes neither training features nor model decisions.

Expanded backend verification: **164tests passed566.55s**, including Claude's
38evaluation tests and seven candidate tests. Seven warnings come from empty
subsets in evaluator metric fixtures; no failures. Retrieval-only independent
development smoke passed replay/causal-prefix/source/public-preservation audits
using two epochs. This is interface verification, not quality acceptance.
Adapter metadata records actual fitting seed/window counts separately from the
schedule so evaluator sample sizes are not mistaken for study defaults.

Independent suite registration:
`ids_ml.experiments.sona_candidate.evaluation_adapter:build_evaluation_adapter`.
Fit mode accepts `architecture=gru|pool|hierarchical|retrieval|gru40`, `representation=raw|relative`, and the predeclared training
recipe keywords plus `mode=fit`. It trains the selected representation on the suite's own training examples,
not on final tests. Saved transfer mode accepts `mode=saved` and `directory=/path/to/candidate` and
retains the saved cap (160), explicitly rejecting oversized inputs. Fit mode uses
a separately declared cap 2048 to accommodate independent long-gap chains; this
does not expand existing saved artifact caps. `prepare`, `score_prepared`,
`window_order`, `describe`, `save`, and class `load` implement Claude's
documented interface. The global validation-selected architecture is the intended
final registration; no architecture will be chosen using sealed-case outcomes.
The adapter also exposes `modules()` for parameter verification and
`flow_features()` for independent prefix-causality checks. `describe()` is
JSON-serializable and reports `parameter_count` and local artifact paths.
Fit-mode inference uses the exact-equivalent optimized preprocessor. Its fitting
statistics and transformed values are identical to the confirmation processor;
this is a computational change, not a new input representation.

## Reproduction commands

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.sona_candidate.study \
  --out-dir data/generated/sona_candidate_confirmation_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.sona_candidate.audit \
  data/generated/sona_candidate_confirmation_v1
.venv/bin/python -m ids_ml.experiments.sona_candidate.summarize \
  data/generated/sona_candidate_confirmation_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.sona_candidate.parameter_control \
  data/generated/sona_candidate_confirmation_v1 \
  data/generated/sona_candidate_parameter_controls_v1
.venv/bin/python -m ids_ml.experiments.sona_candidate.package \
  data/generated/sona_candidate_confirmation_v1 \
  data/generated/sona_candidate_package_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.sona_candidate.retrieval_study \
  data/generated/sona_candidate_confirmation_v1 \
  data/generated/sona_candidate_retrieval_v1
.venv/bin/python -m ids_ml.experiments.sona_candidate.package_final \
  data/generated/sona_candidate_confirmation_v1 \
  data/generated/sona_candidate_retrieval_v1 \
  data/generated/sona_candidate_final_package_v1
.venv/bin/python -m ids_ml.experiments.sona_candidate.predict \
  --candidate data/generated/sona_candidate_confirmation_v1/seed-1301/relative_mixed_gru \
  --telemetry data/generated/sona_candidate_confirmation_v1/seed-1301/familiar.parquet \
  --output /tmp/sona-candidate-predictions.json
```

The study resumes only with matching configuration and implementation hashes.
Each fully trained candidate has its own saved validation results and weights,
so interruption does not discard completed training. The prediction CLI refuses
to overwrite an existing output. Labels are not required for inference. Whole
window scores and their calibration are not online prefix alert guarantees.

## Completed candidate development and audits

The study generated **5,043,301 fresh flows in 54,000 windows**. Thirty-six full
models were trained: eight base arms, two capacity-matched pooling controls and
two final architecture arms, each across three seeds. All 1%, 2% and 5% budgets
remain in 864 per-seed/model/partition rows. Every stage passed replay audits:
192 base prediction checks plus 48 pooling-control and 48 extension checks;
576 + 144 + 144 operating-point checks, respectively. The expanded backend suite
passed 164 tests; final adapter metadata roundtrips passed both architecture
variants in 60.11 seconds. No dependency installs, commits, pushes or public
artifact changes were performed.

Three-seed mean validation recall at the strict budget selected GRU40 (30.21%)
over relative mixed GRU32 (28.17%), relative ordinary GRU32 (28.92%), hierarchical
memory (18.44%) and two-hop retrieval (19.52%). Parameter counts are GRU40 7,642,
retrieval 7,490, relative GRU32 5,346 and matched pooling 5,282. Selection preceded
the extension's test evaluation; neither the test tables nor architecture depth
determined the finalist.

| Held-out condition | GRU40 recall | Actual false alarms |
| --- | ---: | ---: |
| Familiar | 34.83% | 0.72% |
| 60-minute lower count | 31.72% | 0.67% |
| Higher count | 26.33% | 0.33% |
| Duration and count change | 29.78% | 0.56% |
| Inventory-heavy | 26.89% | 1.44% |
| Unseen 45-minute | 33.72% | 1.00% |
| Unseen 90-minute | 32.11% | 0.50% |

These are equal-weight seed means at the 1% validation budget, not production
rates. The lower false-alarm rates versus earlier studies accompany lower recall
and a deliberately different benign calibration mix. Do not attribute that
cross-study difference to architecture or claim the inventory problem is solved.
The matched relative pooling control remains weaker than GRU32, including with
nearly equal parameter counts. Broader training did not reproducibly improve the
relative GRU over ordinary-only training. The hierarchy and retrieval additions
did not meet the predeclared quality gates; preserve both negative results.

**Recommendation: needs further work; no public promotion or six-category
replacement.** The original confirmation and retrieval gates failed. GRU40 is
the strongest validation-selected experimental checkpoint in this batch, with
modest gains and substantial missed attacks. A larger/deeper network did not
establish deeper reasoning. Independent chain tests measure memory directly;
the existing GRU32 counterpart already solves Claude's baseline chains, so that
suite has limited headroom for a new memory superiority claim.

Artifacts:

- Base study: `data/generated/sona_candidate_confirmation_v1/`.
- Capacity controls: `data/generated/sona_candidate_parameter_controls_v1/`.
- Retrieval/GRU40: `data/generated/sona_candidate_retrieval_v1/`.
- All-arm report, gates and 864-row CSV:
  `data/generated/sona_candidate_final_results_v1/`.
- Three optimized, exactly equivalent GRU40 checkpoints:
  `data/generated/sona_candidate_final_package_v1/seed-{1301,1327,1361}/`.

Unified checkpoint inference:
`ids_ml.experiments.sona_candidate.extended_inference.load_predictor(directory)`.
It returns the same label-free prediction interface documented above. Package
manifests include feature-equivalence checks on all eight evaluated partitions,
weight/preprocessor hashes and runtime source hashes. The binary 160-event cap
is retained. Independent fit-mode cap 2048 is a separately trained counterpart,
not an expansion of those saved checkpoints.

Frozen independent registration:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m ids_ml.experiments.sona_evaluation.run \
  --mode final --out-dir data/generated/sona_candidate_independent_final_v1 \
  --candidate ids_ml.experiments.sona_candidate.evaluation_adapter:build_evaluation_adapter \
  --candidate-arg architecture=gru40 --candidate-arg representation=relative \
  --candidate-mode fit
```

Candidate selection is final. No sealed test result will change architecture,
training recipe or thresholds. Next development should investigate information
limits and validation-tail reliability before adding more layers: retain benign
lookalikes, predeclare any new independently observable host/routine context,
and do not manufacture authorization from target labels.

## Final independent assessment

Both sealed runs completed three fresh replicates, with every integrity audit
passed and all candidate/suite/public hashes unchanged at run completion.
Claude's evaluator-owned run is the primary assessment:
`data/generated/sona_evaluation_v1/final/`. The primary agent's additional
replication is `data/generated/sona_candidate_independent_final_v1/`. Both use
the same fixed GRU40/relative registration and the suite's 2% primary validation
budget. No result was used to revise the candidate or select a favorable run.

| Finding | Primary assessment | Additional replication |
| --- | --- | --- |
| Overall gate verdict | Insufficient evidence | Fail |
| Mean recall difference versus retrained GRU32 | -0.15 percentage points | -0.67 percentage points |
| Paired 95% interval for that difference | -0.49 to +0.21 points | -0.96 to -0.37 points |
| Familiar chain triples solved | 100% | 100% |
| Unseen long-gap triples solved | 98.61% | 98.80% |
| Shuffled attack versus swap ordering | 46.85% | 47.69% |

The candidate passes integrity, familiar/lookalike false-alarm comparison, memory
and cost gates. It does **not** pass contextual detection superiority. Its
memory is useful: the Beta and pooled controls score about 50% on same-bag order
pairs, while GRU40 solves them. This does not establish deeper reasoning than
the retrained GRU32, which solves 100% of the familiar and unseen chains.

The critical weakness is shifted false alarms. In the primary long-sparse test,
GRU40 reaches 99.33% recall with **24.58% actual false alarms**, versus GRU32's
96.17% recall and 4.67% false alarms. Dense false alarms are 11.00% versus 6.92%.
Short-window recall also regresses (94.17% versus 99.58%). The additional
replication reproduces the long-sparse weakness (26.83% false alarms), so these
are not grounds for public promotion. The two studies use different generators
and calibration mixes; do not mix these figures with the earlier scan table or
the public six-category Overview metrics.

Beta vote models' very low recall under the predeclared threshold mix partly
reflects their inability to separate benign chain swaps from attacks. The suite
retains a clearly labeled post-hoc non-chain calibration sensitivity analysis;
its numbers cannot replace the gate results or justify an exaggerated Beta
deficit. Saved public checkpoints are transfer tests, while the primary neural
comparison uses retrained counterparts on matched data.

## Measured efficiency and artifact closure

At the reference workload (32 windows, 256 events, two CPU threads), the candidate
uses **46.78 ms** end to end versus the public saved Sona's **57.34 ms**. Model-only
time is 16.51 versus 28.93 ms. Peak inference increments are 18,888 versus 18,944
KiB. The candidate passes the predeclared cost gate, but it is slower end to end
than the retrained GRU32 (42.11 ms) and Beta 2 counterpart (25.63 ms). Do not
claim universal efficiency superiority over Beta models. All 90 timing workload
records are flagged for concurrent CPU load. Cold interpreter/import/load costs
are tens of seconds on the mounted Windows environment and are reported
separately from inference. The suite-trained candidate has 8,002 parameters
because its categorical vocabulary differs from the 7,642-parameter scan model.
See `data/generated/sona_evaluation_v1/efficiency_final.json`.

The optimized processor independently measures **4.77 to 5.83 times faster** than
the original explicit processor on identical 128-window samples across the
three development seeds. Twelve alternating-order paired repeats and three
warmups preserve exact output arrays. This is preprocessing-only, not total
model speed, and concurrent CPU training is disclosed. Raw measurements are
`data/generated/sona_candidate_final_results_v1/preprocessing_timing.json`.

The package verifies all 17 runtime source hashes, three manifest hashes and six
weight/preprocessor hashes. Its CLI produced 1,200 window predictions. All 499
checked public/prior source and artifact files remain unchanged. Scoped
embedding-free indexing added 80 symbols from 17 owned files without pruning
or rebuilding other code. No installs, commits, pushes, deployment or HTML build.

**Final recommendation: preserve the experimental package, retain public Sona,
and prioritize normalization/validation-tail and observable-context research.**
The deeper designs did not win, and larger GRU40 does not demonstrate stronger
overall detection or deeper reasoning than the existing GRU32 architecture.
Six-category retraining and public integration remain deferred because the
predeclared binary improvement and independent superiority requirements failed.
The next draft task is `tasks/sona-validation-tail-context-20261013`. It requires
fresh held-out cases, matched fixed architectures, actual false alarms by length
and subtype, and independently observable context instead of target-derived
intent fields. Closed studies and sealed data remain preserved.
