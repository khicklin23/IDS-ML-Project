# Independent Sona evaluation and efficiency suite

Owner: `claude-sona-eval-20261013`. Task: `tasks/sona-evaluation-20261013` (Git-kb, workspace-only).
Candidate development is owned separately (`src/ids_ml/experiments/sona_candidate/`, `docs/SONA_CANDIDATE.md`). This suite reads candidate files and never edits them.

Status: **complete (2026-10-13).** The protocol was frozen before any candidate result was inspected; amendments are dated in 2.10. Results come only from audited runs. **Verdict: INSUFFICIENT EVIDENCE (evaluator-owned sealed run); FAIL in the candidate-owner replication. Recommendation: do not adopt GRU40 as a Sona replacement (section 8).**

All results here come from a Python generator. They describe behavior on constructed synthetic telemetry, not real-world intrusion-detection accuracy. "Reasoning" is used only for observable behavior: whether a model combines evidence across a window, across distance, and in the correct order.

## 1. Questions

1. **Contextual detection.** Does the candidate find attack activity inside mixed host windows at fixed false-alarm budgets, including under changed duration, density, pace and timing noise?
2. **Sequence memory.** Does it use the order of events, and does it retain early evidence across increasing gaps and distractor events? Is this more than finding a single obvious flow?
3. **Generalization.** How does it behave on durations, paces and gaps absent from training?
4. **False alarms on benign lookalikes.** Inventory sweeps, periodic polling, service-outage retries and user login mistakes.
5. **Efficiency.** Parameters, checkpoint size, cold and warm latency, throughput and peak memory on identical settings, with preprocessing timed separately from the model.

## 2. Protocol `sona-eval-1.0.0` (frozen)

The machine-readable protocol is `src/ids_ml/experiments/sona_evaluation/protocol.py`. Every run writes `protocol.json` and its SHA-256, plus SHA-256 of every suite source file. A run refuses to resume if either changes.

### 2.1 Units and targets

The evaluation unit is a **window**: one host's chronologically ordered flows. A window's target is 1 if it contains any generated attack component, else 0. Each flow also carries a component-level generator label (`attack_flow`), used only as a training target for flow-level counterparts and for flow-unit reporting. Labels, component names, pace, condition and triple membership are kept in a separate key table and are **never passed to an adapter at inference**. Window and flow identifiers are random hexadecimal strings, so they encode nothing.

### 2.2 Generated families

Background traffic: each window draws an environment service mix (HTTPS, HTTP, DNS, SSH, NTP, SMTP) and a background rate. Most flows are established. There is a small rate of isolated failures, so a single failed connection is not evidence of an attack.

| Component | Target | Pace variants | Distinguishing observable (by design) |
| --- | ---: | --- | --- |
| `scan` | 1 | ordinary (10 to 40 distinct ports in 20 to 180 s), slow (span 30 to 80% of window) | mostly rejected or timed-out, distinct ports, random port order |
| `bruteforce` | 1 | ordinary (10 to 40 attempts in 30 to 300 s), slow | many short reset/rejected attempts to one auth port (SSH/RDP) |
| `beacon` | 1 | ordinary (30 to 90 s interval, jitter 0.1), slow (150 to 400 s, jitter 0.25) | periodic, near-constant small payloads to an uncommon port |
| `inventory` | 0 | span 20 to 90% of window | distinct ports in **ascending** order, mostly established |
| `polling` | 0 | interval 30 to 400 s | periodic, but to HTTPS/HTTP with large variable responses |
| `retries` | 0 | 1 to 3 bursts | timeouts to one common web/DNS service (outage) |
| `user_auth` | 0 | one episode | 1 to 4 failed logins then success (mistyped password) |

**Intrinsic ambiguity, stated explicitly.** Scan and inventory differ only in failure rate and port order. These are sampled from separated distributions, but with few probes the realized failure fractions can overlap. Some scan-versus-inventory errors are therefore expected even for an ideal detector, and the suite reports them by subtype. The suite never adds authorization or intent metadata derived from the target. Beacon versus polling is distinguishable per flow (port and payload), so those cases test false alarms, not memory.

### 2.3 Conditions

Train and validation use the **familiar mixture**: window duration T = 1,800 s; background rate 1.6 to 2.8 flows/min; ordinary and slow attacks 50/50; 30% of attack windows also contain a benign lookalike; benign windows are 40% plain background and 15% each lookalike. Chain triples (2.4) are included.

| Test condition | Change from familiar | Seen in training? |
| --- | --- | --- |
| `familiar` | none (fresh samples) | yes |
| `slower` | attacks slower than any training pace (spans 85 to 98% of T, fewer probes, beacon 400 to 550 s) | no |
| `long_sparse` | T = 7,200 s, same rates (longer sequences, sparser evidence) | no |
| `short` | T = 600 s | no |
| `dense` | background 4 to 7 flows/min | no |
| `sparse` | background 0.5 to 0.9 flows/min | no |
| `noisy_timing` | periodic jitter x3, 10% missed beacons, 5 s timestamp noise | no |
| `lookalike_heavy` | every benign window holds a lookalike, 25% each | partially |
| `mixed` | every window holds a lookalike, attack windows hold one attack too | partially |

### 2.4 Order and memory: chain triples

A compromise chain is generated in triples that share pre-background, distractors and post-background exactly:

```text
attack     : [pre] [F x k] [D distractors] [S T] [post]   target 1
swap       : [pre] [S T] [D distractors] [F x k] [post]   target 0  (identical flow multiset to attack)
innocent   : [pre] [E x k] [D distractors] [S T] [post]   target 0  (identical suffix to attack)
```

`F`: failed logins to one auth port. `S`: successful login to that port. `T`: large outbound transfer. `E`: routine successful logins with the same timing distribution as `F`. Flows keep their own `prior_gap_s` when blocks are reordered, so attack and swap have identical per-flow measurements and differ only in order. The label is a generator definition:

- **Attack:** repeated failures, then success, then a large transfer.
- **Swap (benign by definition):** a completed session, followed later by a stale-credential job failing. In a real network an analyst might still review this.

What this design rules out:

- An **order-invariant** model (voting, pooling, bag-of-flows) must score attack and swap identically.
- A **suffix-only** model must score attack and innocent identically once D is large enough.
- A **single-flow** detector can separate attack from innocent (the `F` flows) but not attack from swap.

Only a model that relates the early block to the late block, across D distractors, solves the whole triple. In half of the triples, D includes two or three isolated decoy failures (identical across the triple), so one failed flow is not sufficient evidence.

- Familiar gaps: D in {0, 2, 4, 8, 16, 32}. These appear in training.
- Unseen gaps (`chain_long`): D in {64, 128, 256}. These are not in training.

Metrics are threshold-free and paired within a triple. Ties count one half.

- `attack>swap`: the fraction of triples where the attack window scores higher than its swap.
- `attack>innocent`: the same against the innocent window.
- `triple_solved`: both comparisons hold at once.

### 2.5 Controls

- **Shuffled interior:** flows other than the first and last are permuted within the window, with event times recomputed from carried gaps. Used on `familiar` and on chain triples. This removes order while keeping the measurement multiset.
- **Truncated history:** only the last K in {8, 32} flows are kept. This shows whether a decision depends on recent context only. On chains, a high `attack>swap` under K = 8 means the order decision is local, not long-range memory.
- **Order-invariant and single-flow comparators** (2.6) bound what is achievable without order.

### 2.6 Models compared

Retrained counterparts are trained per replicate on the suite's own training partition. Checkpoints are selected by validation loss only.

| Key | Kind | Description |
| --- | --- | --- |
| `beta1_counterpart` | retrained | Beta 1/2 MLP (32-16) on flows, component labels, trained **without slow-pace attack windows**; window score = fraction of flows predicted attack (vote) |
| `beta2_counterpart` | retrained | same architecture and vote, full training mixture |
| `sona_counterpart` | retrained | public Sona architecture: 1-layer GRU, hidden 32, absolute timing preprocessing, window targets |
| `sona_shuffled_trained` | retrained control | same GRU trained and evaluated on interior-shuffled windows |
| `pooled_mlp` | retrained control | per-flow MLP, mean-pooled logits, window targets (order-invariant neural) |
| `summary_boosting` | retrained | gradient boosting on engineered window summaries, including order features (ascending-port fraction, periodicity) |
| `temporal_rule` | rule | deterministic comparison heuristic, never a primary detector |
| `beta1_saved`, `beta2_saved` | saved checkpoint | `data/generated/model_versions/v1`, `v2`; six-class flow MLP; score = fraction of flows not predicted benign |
| `sona_public_saved` | saved checkpoint | `data/generated/showcase_v4/artifact_sequence`; six-class GRU; score = 1 - P(benign) |
| candidate | adapter | see 3; fit-mode is primary, saved-mode is reported as transfer |

Saved checkpoints were trained on the six-family session generator, not on these windows. Their results are **zero-shot transfer**, labeled as such, and are never mixed with retrained counterparts in one unlabeled number.

### 2.7 Thresholds and metrics

- **Thresholds:** for each model, thresholds are selected on benign validation windows at false-alarm budgets of 1%, 2% and 5%, strictly above the next rejected score. They are frozen and written to disk with a hash before any test window is scored. The primary operating point is **2%**.
- **Window units (primary):** recall, **actual** test false-positive rate, precision and F1 at each frozen threshold, plus ROC AUC and average precision.
- **Flow units (secondary):** the fraction of attack-component flows inside flagged windows, and the flow-weighted benign false-alarm rate.
- **Subtypes:** per-family and pace flag rates, and per-lookalike false-alarm rates.
- **Seeds:** per-replicate values with min/max. Means are across replicates, never pooled across conditions without saying so.
- **Paired comparisons:** model A minus model B on identical windows. A 95% interval comes from a paired bootstrap that resamples windows within each replicate (2,000 resamples, fixed seed).

### 2.8 Acceptance criteria (predeclared)

The primary comparison is **candidate (fit-mode) versus `sona_counterpart`**, on sealed final cases with at least 3 replicates, at the 2% budget. The Beta counterparts and saved checkpoints are reported for every gate, but they do not gate.

| Gate | Requirement |
| --- | --- |
| G0 Integrity | Adapter contract checks; inputs contain no label/provenance columns; saved-prediction replay within 1e-6; candidate source and artifact hashes recorded before sealed generation and unchanged after; suite and public artifact hashes unchanged; disjoint identities; capture bounds; train-only preprocessing for suite counterparts; any declared causal feature claim verified by prefix invariance. |
| G1 Contextual detection | Macro-mean over the 9 conditions of the recall difference has a bootstrap 95% lower bound > 0, and no single condition has an upper bound < -5 points. |
| G2 False alarms | On `familiar` and `lookalike_heavy`, the actual false-positive-rate difference has a 95% lower bound <= +1 point. |
| G3 Memory | On familiar-gap chains: `attack>swap`, `attack>innocent` and `triple_solved` are each >= 0.80. On unseen gaps: `triple_solved` >= 0.70. The shuffled-interior control lowers `attack>swap` to <= 0.65. Familiar `triple_solved` exceeds `pooled_mlp` with a paired 95% lower bound > 0. |
| G4 Cost | Warm median latency (batch 32, length 256, 2 threads) and peak-RSS increment are each <= 3x `sona_public_saved`; **or** the G1 macro recall gain is >= 5 points. |

Recommendation rules:

- **PASS:** G0 to G4 all hold.
- **FAIL:** G0 fails; or G1 has a macro upper bound < 0 or any condition below -5 points; or G2 fails; or G4 fails while G1 passes.
- **INSUFFICIENT EVIDENCE:** everything else. This includes no compatible candidate, no fit-mode support, fewer than 3 sealed replicates, an inconclusive G1, or memory not demonstrated.

A higher threshold budget is a tradeoff, not a stronger model. Thresholds are never chosen on test windows. Final-case outcomes are never used to select, tune or retrain any model.

### 2.9 Sealing

Development fixtures and the baseline study use declared seeds (below). The candidate developer may use them. **Final cases do not exist until the final run.** Their seeds are drawn from the OS random source only after the candidate's source tree and artifacts have been hashed and recorded. They are written into the final results for reproduction. Final cases use a separate identifier namespace. This guards against tuning on the final cases. It cannot prevent a deliberately adversarial process on the same machine, which is a stated limitation.

| Purpose | Seeds |
| --- | --- |
| Development fixtures | 52001 |
| Baseline replicates (train/validation/test/model) | 52011, 52037, 52069 |
| Final replicates | drawn at final-run time (recorded) |

These are disjoint from every seed used by earlier studies (7, 1007, 11 to 101, 451 to 577, 701 to 769, 811 to 853, 941 to 983, 1103 to 1151).

### 2.10 Amendments (2026-10-13; items 1 to 3 before any result was inspected, items 4 and 5 after the baseline and before any candidate result)

1. **Ties in paired triple comparisons.** Scores are rounded to 1e-6 before attack/swap/innocent comparisons, so floating-point summation-order noise in order-invariant models counts as a tie (one half), as section 2.4 intends. `triple_solved` is the product of the two comparison credits. This is implemented in the summarizer from saved scores; run-time `metrics.json` keeps the strict values.
2. **G4 definitions.**
   - "Latency" means end-to-end warm median for one batch (`prepare` plus `score_prepared`), at batch 32, length 256 and 2 threads. Model-only latency is reported alongside.
   - The memory ratio compares peak-RSS increments over the post-load baseline in a dedicated fresh process, with a 16 MiB noise floor applied to both sides.
   - Cold-start repetitions run at the reference thread count only.
3. **Run-source hashing.** The resume and audit hash covers the files that determine cases, scores, thresholds and audits: `__init__`, `protocol`, `cases`, `metrics`, `adapters`, `audit`, `run`. Reporting and efficiency scripts post-process saved outputs and are excluded.
4. **Efficiency corrections, made after the baseline run and before any candidate result.**
   - **Peak memory.** The first efficiency pass (`efficiency_baseline.json`, superseded but kept) measured `ru_maxrss`. That figure was dominated by library import and parquet reading, so every inference increment read 0. Peak memory is now isolated by resetting `VmHWM` through `/proc/self/clear_refs` immediately before each configuration's calls.
   - **Boosting adapter.** `SummaryBoosting.score_prepared` entered `threadpool_limits` on every call, adding about 0.25 s of library inspection to each call. That wrapper is removed. This is performance only: re-scoring the saved baseline models on `familiar`, `dense` and `chain_long` for all three replicates reproduced the saved scores exactly (maximum difference 0.0). Because `adapters.py` is a run-source file, the baseline run directory's stamped hash predates this change. Resuming the baseline would be refused, as designed. The final run stamps the corrected source.
5. **Candidate freeze scope (before any candidate result).** The final-run freeze also hashes `docs/SONA_CANDIDATE.md`. The candidate owner may append results to that document while the sealed run is in progress. G0 therefore requires unchanged candidate source and artifacts, and reports any document-only change separately rather than treating it as a code change.

## 3. Adapter contract

Code: `src/ids_ml/experiments/sona_evaluation/adapters.py`.

```python
class EvaluationAdapter:
    name: str
    provenance: str            # 'saved_checkpoint' or 'retrained'
    causal_features: bool      # claim; verified by prefix invariance if True
    def fit(self, train, val, seed) -> None             # optional; retrained mode only
    def prepare(self, flows) -> object                  # preprocessing only, timed separately
    def score_prepared(self, prepared) -> np.ndarray    # model forward only
    def window_order(self, prepared) -> list[str]       # window ids aligned with scores
    def describe(self) -> dict                          # parameters, modules, artifact_paths, notes
    def save(self, directory) / load(directory)         # retrained mode: replay audit
```

- **Inference input:** a label-free DataFrame with exactly the allowed columns: the 11 schema feature columns, `session_id` (opaque window id), `event_time` and `flow_id`. Rows are chronological within each window and windows are in random order.
- **Score:** a finite float per window. Higher means more suspicious. It need not be a probability.
- **Training input (`fit`):** train and validation frames that also carry `window_target` and `attack_flow`. Test frames never reach `fit`.
- **Registration:** a candidate exposes `build_evaluation_adapter(**kwargs)` in a module under `ids_ml.experiments.sona_candidate`. The suite loads it with `--candidate module:function` and optional `--candidate-arg key=value`.

## 4. Efficiency protocol

Command: `python -m ids_ml.experiments.sona_evaluation.efficiency`.

- **Isolation:** every configuration runs in a fresh subprocess with `torch.set_num_threads(t)`, matching `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, and `threadpoolctl` limits.
- **Inputs:** benchmark windows come from a development fixture, built at exact lengths L in {64, 256, 1024} and batch sizes B in {1, 32, 256}. These are synthetic inputs.
- **Measured quantities:**
  - parameter count, from `describe()` modules;
  - checkpoint bytes;
  - preprocessing time and model time, measured separately;
  - warm latency: 3 warmup calls, then 20 timed repeats, reporting median and p90;
  - cold latency: fresh process, import plus load plus the first call, 5 repetitions;
  - throughput: windows/s and flows/s;
  - peak RSS (`ru_maxrss`) and its increment over the post-import baseline.
- **Machine:** CPU model, logical cores and library versions are recorded, plus 1-minute load average before and after each measurement. Measurements with load above 1.0 are flagged, because another session may be training on the same machine.

## 5. Commands

All commands run from the repository root with the existing `.venv`. Outputs go under the ignored `data/generated/sona_evaluation_v1/`. Every stage is resumable: rerunning the same command skips completed stages, and refuses if the protocol or run-source hashes changed.

```bash
# Regression tests (38 tests)
.venv/bin/python -m pytest tests/sona_evaluation -q

# Development smoke (seed 52001, tiny sizes, 3 epochs)
.venv/bin/python -m ids_ml.experiments.sona_evaluation.run --out-dir /tmp/sona-eval-smoke --mode smoke

# Bounded baseline study (seeds 52011/52037/52069; ~17 min per replicate on this machine)
.venv/bin/python -m ids_ml.experiments.sona_evaluation.run \
  --out-dir data/generated/sona_evaluation_v1/baseline --mode baseline
.venv/bin/python -m ids_ml.experiments.sona_evaluation.summarize data/generated/sona_evaluation_v1/baseline \
  --compare sona_counterpart beta1_counterpart --compare sona_counterpart beta2_counterpart \
  --compare sona_counterpart pooled_mlp --compare sona_counterpart summary_boosting

# Efficiency (fresh subprocess per adapter and thread count, CPUs 8-11 via taskset)
.venv/bin/python -m ids_ml.experiments.sona_evaluation.efficiency \
  --out data/generated/sona_evaluation_v1/efficiency_baseline_v2.json \
  --replicate-dir data/generated/sona_evaluation_v1/baseline/replicate-52011

# Final sealed candidate evaluation, as run (candidate frozen first; seeds drawn at run time)
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python -m ids_ml.experiments.sona_evaluation.run --mode final \
  --out-dir data/generated/sona_evaluation_v1/final \
  --candidate ids_ml.experiments.sona_candidate.evaluation_adapter:build_evaluation_adapter \
  --candidate-arg architecture=gru40 --candidate-arg representation=relative --candidate-mode fit
.venv/bin/python -m ids_ml.experiments.sona_evaluation.efficiency \
  --out data/generated/sona_evaluation_v1/efficiency_final.json \
  --replicate-dir data/generated/sona_evaluation_v1/final/replicate-518997229 \
  --candidate ids_ml.experiments.sona_candidate.evaluation_adapter:build_evaluation_adapter \
  --only candidate_fit sona_public_saved sona_counterpart beta2_counterpart pooled_mlp
.venv/bin/python -m ids_ml.experiments.sona_evaluation.summarize data/generated/sona_evaluation_v1/final \
  --candidate candidate_fit --reference sona_counterpart \
  --efficiency data/generated/sona_evaluation_v1/efficiency_final.json \
  --compare sona_counterpart beta2_counterpart --compare candidate_fit summary_boosting --compare candidate_fit pooled_mlp
```

## 6. Results

### 6.1 Baseline study (audited, 3 replicates, no candidate)

- **Run:** `data/generated/sona_evaluation_v1/baseline`, protocol hash `eba31b187326…`. Full tables are in `RESULTS.md` and `summary.json` there.
- **Scale:** each replicate holds 3,000 training windows, 3,000 validation windows, 7,200 condition-test windows, 1,080 chain-test windows and 6 control sets.
- **Audits:** all three replicate audits passed. That covers data hashes, disjoint identities, capture bounds and order, label-column refusal by every adapter, train-only preprocessing, frozen thresholds, exact save/load prediction replay, prefix causality, and suite-source and public-artifact preservation (1,160 files checked). The run-level audit reports public artifacts and suite source unchanged.

**Recall / actual false-positive rate at the frozen 2% validation budget**, as means across 3 replicates in %, window units. Per-seed ranges, the 1% and 5% budgets, precision, F1 and flow units are in `RESULTS.md`.

| Condition | Beta 1 cpt. | Beta 2 cpt. | Sona cpt. | Sona shuffled-trained | Pooled MLP | Summary boosting | Rule | Beta 2 saved | Public Sona saved |
|---|---|---|---|---|---|---|---|---|---|
| familiar | 0.0 / 0.0 | 0.2 / 0.0 | 97.9 / 3.0 | 92.9 / 1.8 | 49.7 / 0.6 | 80.6 / 3.2 | 35.6 / 2.6 | 0.0 / 0.0 | 0.1 / 0.8 |
| slower | 0.0 / 0.0 | 0.0 / 0.0 | 88.6 / 2.3 | 78.3 / 1.2 | 3.1 / 0.7 | 54.6 / 2.3 | 12.2 / 3.2 | 0.0 / 0.0 | 0.5 / 1.3 |
| long_sparse | 0.0 / 0.0 | 0.0 / 0.0 | 94.8 / 4.5 | 87.2 / 11.5 | 16.0 / 0.0 | 74.8 / 6.6 | 40.7 / 3.9 | 0.0 / 0.0 | 0.2 / 0.7 |
| short | 29.5 / 3.9 | 31.9 / 7.3 | 98.1 / 5.1 | 76.4 / 2.1 | 79.9 / 6.2 | 73.0 / 6.9 | 33.8 / 2.5 | 12.0 / 0.0 | 0.1 / 1.0 |
| dense | 0.0 / 0.0 | 0.0 / 0.0 | 97.7 / 7.6 | 94.8 / 13.8 | 11.2 / 0.0 | 81.4 / 10.2 | 30.3 / 3.6 | 0.0 / 0.0 | 0.3 / 0.5 |
| sparse | 34.2 / 1.2 | 40.8 / 6.3 | 98.4 / 4.5 | 92.7 / 3.8 | 79.7 / 4.0 | 73.2 / 2.8 | 44.5 / 2.7 | 12.6 / 0.0 | 0.2 / 0.8 |
| noisy_timing | 0.2 / 0.0 | 0.6 / 0.0 | 98.1 / 3.8 | 91.8 / 2.2 | 49.2 / 0.6 | 80.4 / 3.2 | 31.4 / 2.6 | 0.1 / 0.0 | 0.0 / 0.5 |
| lookalike_heavy | 0.1 / 0.0 | 0.8 / 0.0 | 97.7 / 5.8 | 93.3 / 2.5 | 48.5 / 1.4 | 81.6 / 3.8 | 35.6 / 5.3 | 0.0 / 0.0 | 0.0 / 1.0 |
| mixed | 0.2 / 0.0 | 1.4 / 0.0 | 94.2 / 4.7 | 86.2 / 4.0 | 51.0 / 0.5 | 76.0 / 3.6 | 37.8 / 4.9 | 0.0 / 0.0 | 0.3 / 0.8 |

**Mean ROC AUC on `familiar` (threshold-free ranking):**

| Model | ROC AUC |
|---|---|
| Sona counterpart | 0.997 |
| Sona shuffled-trained | 0.995 |
| Pooled MLP | 0.968 |
| Summary boosting | 0.944 |
| Beta 1 counterpart | 0.937 |
| Beta 2 counterpart | 0.920 |
| Beta 1 saved | 0.820 |
| Beta 2 saved | 0.784 |
| Public Sona saved | 0.479 |

**Memory (chain triples, paired, threshold-free, %).** Familiar gaps 0 to 32; unseen gaps 64 to 256.

| Model | attack>swap | attack>innocent | triple solved (familiar) | triple solved (unseen) | shuffled attack>swap | last-8 attack>swap | last-32 triple solved |
|---|---|---|---|---|---|---|---|
| Sona counterpart | 100.0 | 100.0 | 100.0 | 100.0 | 51.0 | 6.9 | 84.6 |
| Sona shuffled-trained | 52.0 | 100.0 | 52.0 | 50.0 | 48.7 | 0.6 | 38.0 |
| Pooled MLP | 50.2 | 100.0 | 50.2 | 49.6 | 50.1 | 0.0 | 37.4 |
| Beta 1 / Beta 2 cpt. | 50.0 | 100.0 | 50.0 | 50.0 | 50.0 | 16.5 / 15.8 | 40.2 |
| Summary boosting | 100.0 | 100.0 | 100.0 | 93.7 (gap 256: 82.2) | 51.7 | 0.0 | 83.9 |
| Temporal rule | 50.5 | 90.7 | 45.9 | 38.3 | 49.3 | 12.3 | 40.9 |
| Public Sona saved | 28.5 | 46.3 | 8.5 | 26.9 | 49.6 | 93.7 | 10.3 |

### 6.2 Baseline interpretation

**1. The Sona architecture shows a demonstrated memory benefit.** Retrained on the suite's windows, the public Sona architecture (GRU 32) solves every triple at every gap, including unseen gaps up to 256 distractors.

- The order-invariant models score attack and swap identically. A regression test proves this structurally, and their attack>swap is exactly 50%.
- The memory is removed by order destruction: the shuffled-input control gives 51%, and the GRU trained on shuffled windows gives 52%.
- The memory is removed by losing early evidence: keeping only the last 8 flows drops attack>swap to 7%. The model then scores the swap's late failure burst as more suspicious, so it was not relying on the suffix.

This is the behavior "memory" should mean here: relating an early block to a late block across distractors, in the correct order.

**2. Memory is not unique to neural recurrence.** Engineered summaries with generic order features also solve the chains: failure-time centroid, position of the largest transfer, same-port failure runs. Summary boosting scores 100% at familiar gaps but degrades at unseen gap 256 (82%). It is 16 to 34 points lower than the Sona counterpart on general detection.

**3. On contextual detection, the GRU leads everywhere.** It has the highest recall in every condition (88.6 to 98.4%). The paired bootstrap gives +21 points macro recall over summary boosting, with a 95% interval of [20.2, 21.9], and better recall in every condition. The weaknesses:

- **Benign inventory sweeps:** 15 to 21% are flagged. This is the documented scan/inventory ambiguity.
- **False alarms above budget under unseen conditions:** actual rates reach 7.6% (`dense`), 5.8% (`lookalike_heavy`) and 5.1% (`short`) against a 2% validation budget. A frozen threshold does not transfer its false-alarm rate to shifted traffic.
- **Unseen slower attacks:** recall falls to 88.6%. Slower brute force is the weakest subtype at 73.7%.

**4. The Beta counterparts' near-zero recall at the frozen thresholds is partly a protocol interaction, not only weak detection.** Validation benigns include chain swaps whose flows match attack chains. Vote models cannot separate those, so they score them high, which raises the frozen threshold. This was predeclared and is kept. A post-hoc sensitivity analysis, not a gate, puts thresholds on non-chain validation benigns. It gives, on `familiar`:

| Model | Recall | Actual FPR |
|---|---|---|
| Beta 1 counterpart | 36.8% | 2.4% |
| Beta 2 counterpart | 19.0% | 2.2% |
| Saved Beta 2 | 62.0% | 2.4% |
| Sona counterpart | 97.7% | 2.3% |

The ranking conclusion is unchanged. The size of the Beta deficit in the predeclared table is overstated by this coupling.

**5. The public Sona checkpoint does not transfer to these windows.** Its AUC is 0.43 to 0.49 (about chance). The saved Beta checkpoints transfer partially (AUC 0.75 to 0.83). The public six-class session GRU was trained on single-scenario sessions and has not learned mixed-window detection. Its zero-shot numbers say nothing about the architecture, which the retrained counterpart measures.

**6. Headroom.** The Sona counterpart is close to the recall ceiling on most conditions. A candidate can show a G1 gain mainly on `slower`, `long_sparse` and `mixed`, and on false-alarm rates. Where results are near the ceiling, the suite separates designs less than it would with harder cases. This is a limitation of this suite, not evidence for or against any candidate.

**7. Uncertainty.** Bootstrap intervals resample windows within each replicate. They do not include training-seed variance; per-seed min/max ranges are reported for that.

### 6.3 Candidate evaluation: sealed final run (evaluator-owned)

**Candidate frozen by its owner.** GRU with 40 hidden units over relative-timing inputs (`architecture=gru40, representation=relative`). It was selected by the candidate owner on three-seed validation recall only, and was registered with `ids_ml.experiments.sona_candidate.evaluation_adapter:build_evaluation_adapter`.

**Fit mode is the primary evaluation.** The candidate's own recipe is retrained on the suite's training partition with the replicate seed. It has 8,002 parameters with this suite's input vocabulary; the packaged checkpoints have 7,642.

**Saved-transfer mode was not evaluable.** The packaged checkpoints keep a 160-flow cap and explicitly reject longer windows, and `long_sparse`, `dense` and `chain_long` routinely exceed it. This is reported as an incompatibility rather than scoring a filtered subset.

**Run and audits.**

- **Run:** `data/generated/sona_evaluation_v1/final`. Full tables are in `RESULTS.md` and `summary.json` there.
- **Sealing:** candidate source (17 files) and its document were hashed at 14:14:04.193 UTC. Sealed seeds 518997229, 831816426 and 885659123 were drawn from the OS 9 ms later.
- **Audits:** all three replicate audits passed, including the candidate's exact save/load replay and its prefix-causality claim. The run-level audit reports public artifacts, suite source, and candidate source and artifacts unchanged after evaluation.

**Contextual detection at the frozen 2% budget (recall / actual false-positive rate, %, means of 3 sealed replicates).** Per-seed ranges, the 1% and 5% budgets and all other models are in `RESULTS.md`.

| Condition | Candidate GRU40 | Sona counterpart (GRU32) | Paired recall difference [95% CI] | Paired FPR difference [95% CI] | Reading |
|---|---|---|---|---|---|
| familiar | 98.1 / 2.3 | 98.5 / 2.8 | -0.4 [-1.3, 0.4] | -0.4 [-1.3, 0.4] | no demonstrated difference |
| slower | 86.7 / 2.1 | 85.3 / 2.5 | +1.3 [-0.2, 2.9] | -0.4 [-1.3, 0.3] | no demonstrated difference |
| long_sparse | 99.3 / **24.6** | 96.2 / 4.7 | +3.2 [2.2, 4.2] | **+19.9 [17.7, 22.2]** | recall bought with many false alarms |
| short | 94.2 / 3.2 | 99.6 / 5.0 | **-5.4 [-6.7, -4.1]** | -1.8 [-2.8, -1.0] | worse recall |
| dense | 99.2 / 11.0 | 98.8 / 6.9 | +0.5 [-0.2, 1.2] | +4.1 [2.6, 5.7] | more false alarms |
| sparse | 98.1 / 3.8 | 98.4 / 4.5 | -0.3 [-1.2, 0.5] | -0.7 [-1.6, 0.3] | no demonstrated difference |
| noisy_timing | 98.5 / 2.7 | 98.6 / 2.7 | -0.1 [-1.0, 0.8] | +0.1 [-0.8, 0.9] | no demonstrated difference |
| lookalike_heavy | 98.8 / 4.4 | 98.6 / 5.2 | +0.3 [-0.4, 0.9] | -0.8 [-1.9, 0.3] | no demonstrated difference |
| mixed | 96.7 / 3.3 | 97.0 / 3.3 | -0.3 [-1.4, 0.7] | -0.0 [-1.0, 1.0] | no demonstrated difference |

Macro recall difference: -0.15 points, 95% CI [-0.49, +0.21].

**Further detail:**

- **Per-seed ranges:** the candidate's `long_sparse` false-alarm rate is 14.0 to 34.8% across seeds; the Sona counterpart's is 3.5 to 6.8%. The candidate's `short` recall is 88.0 to 98.0%; the Sona counterpart's is 99.5 to 99.8%.
- **Other budgets:** the `long_sparse` false-alarm rate stays at 23.2 / 24.6 / 25.8% at the 1 / 2 / 5% budgets. It is a distribution shift, not a threshold choice.
- **Precision and F1 on `familiar`:** precision 97.7%, F1 97.9%, the same as the Sona counterpart's F1 of 97.9%.
- **Flow units:** 98.4% of attack flows fall in flagged windows (Sona counterpart 99.0%), and 2.7% of benign flows inherit a flag (Sona counterpart 3.4%).
- **Subtypes:**
  - **Inventory sweeps:** 12.3% of benign inventory windows are flagged on `familiar`, against 16.0% for the Sona counterpart. The ambiguity remains for both.
  - **Slower brute force:** 71.2% recall versus 63.1%.
  - Other families are within about 2 points.

**Why the candidate fails on long, dense and short windows.** This is a measured explanation, not a guess.

- **Benign false alarms rise with window length:**
  - In every seed, the candidate's `long_sparse` benign false-alarm rate climbs steadily from the shortest to the longest quarter of windows: 9 to 17% to 24 to 61%. The Sona counterpart's stays at 1 to 12%.
  - It affects every benign type, including plain background: 15.5% versus 0.2%.
- **The cause is an input outside the training range.** The candidate's relative "elapsed units" input (elapsed time divided by the mean of earlier gaps) grows with the number of events, so it is not scale-free. Its standardized median is 0.25 in training (99th percentile 1.32), but 1.59 on `long_sparse`, 1.11 on `dense` and -0.46 on `short`. More than half of the long-window flows lie beyond the training range's 99th percentile.
- **This one feature accounts for all three weaknesses:** false alarms on long and dense windows, and lost recall on short ones. The public Sona architecture uses absolute gaps and does not show it.
- **History:** this repeats the duration-extrapolation failure recorded for pace features in `docs/SCAN_CONTEXT_BENCHMARK.md`.

**Memory (chain triples, paired, threshold-free, %; mean (min to max) over 3 sealed replicates).**

| Model | attack>swap | attack>innocent | triple solved, familiar gaps | triple solved, unseen gaps 64 / 128 / 256 | shuffled attack>swap | last-8 attack>swap | last-32 triple solved |
|---|---|---|---|---|---|---|---|
| Candidate GRU40 | 100.0 | 100.0 | 100.0 | 100.0 / 99.7 / 96.0 | 46.9 | 35.7 (14.4 to 56.7) | 85.4 |
| Sona counterpart GRU32 | 100.0 | 100.0 | 100.0 | 99.4 / 100.0 / 97.4 | 48.6 | 5.2 | 85.2 |
| Summary boosting | 100.0 | 100.0 | 100.0 | 96.5 / 90.6 / 79.1 | 49.4 | 0.2 | 84.7 |
| Pooled MLP (order-invariant) | 50.0 | 100.0 | 50.0 | 50.5 overall | 49.4 | 0.0 | 37.9 |
| GRU32 trained on shuffled windows | 52.2 | 100.0 | 52.2 | 48.0 overall | 48.0 | 1.1 | 40.0 |

The candidate demonstrates order- and gap-dependent memory by every predeclared test:

- It solves 100% of familiar-gap triples and 98.6% of unseen-gap triples.
- Shuffling drops attack>swap to 47%.
- It beats the order-invariant pooled model by 50 points (95% CI [49.4, 50.6]).

**But it shows no memory benefit over the existing Sona architecture.** The difference in triple-solved rate is 0.0 at familiar gaps and -0.3 [-1.4, 0.8] at unseen gaps.

Under last-8 truncation the candidate still prefers attack over swap 35.7% of the time, against 5.2% for GRU32. So when early evidence is removed it relies somewhat more on the local suffix, with large variation across seeds. This is diagnostic only, not a gate.

**Efficiency.** Source: `data/generated/sona_evaluation_v1/efficiency_final.json`. Models from sealed replicate 518997229, all measured in one batch on CPUs 8 to 11 with 2 threads. Load average was 4 to 5 during measurement because Codex's concurrent sealed run was training, so all rows are flagged.

| Model | Parameters | Checkpoint bytes | Batch 32 × 256 flows: preprocess + model = total (ms) | Batch 1 × 64 flows (ms) | Throughput, 256 × 1,024 flows (windows/s) | Inference peak RSS increment, reference / 256 × 1,024 |
|---|---|---|---|---|---|---|
| Candidate GRU40 | 8,002 | 46,435 | 27.7 + 16.5 = 46.8 | 15.2 | 328 | 8.1 / 228 MiB |
| Sona counterpart GRU32 | 5,538 | 39,542 | 22.0 + 20.1 = 42.1 | 17.2 | 350 | 7.4 / 196 MiB |
| Public Sona (saved) | 5,670 | 27,996 | 26.8 + 28.9 = 57.3 | 14.2 | 376 | 7.4 / 196 MiB |
| Beta 2 counterpart | 1,330 | 22,746 | 22.4 + 2.8 = 25.6 | 11.9 | 644 | 4.8 / 129 MiB |
| Pooled MLP | 1,330 | 16,447 | 20.8 + 2.1 = 23.3 | 11.0 | 785 | 5.0 / 126 MiB |

- **Cold start** (fresh process to first scored batch) is 41 to 51 s for every model. That is mostly importing PyTorch, pandas and scikit-learn from the Windows-mounted virtual environment (adapter load 9 to 12 s), so it is an environment cost, not a model difference.
- **Noise.** Measurements on a quiet repeat differ by up to about 30% between identical architectures; the baseline v2 pass measured public Sona at 33.4 ms and the Sona counterpart at 23.7 ms. Ratios below 2x are therefore not meaningful.
- **G4 ratios against public Sona at the reference configuration:** latency 0.82, model-only 0.57, memory 1.00, so G4 passes.
- **Direct comparison with the Sona counterpart:** 1.11x total latency, 1.44x parameters. The candidate's model step is cheaper; its relative-timing preprocessing costs more.

### 6.4 Gate outcome (predeclared rules)

| Gate | Result | Basis |
|---|---|---|
| G0 Integrity | pass | all replicate audits and run-level preservation checks |
| G1 Contextual detection | **not met** | macro recall difference -0.15 [-0.49, +0.21] (lower bound not > 0); worst condition `short`, upper bound -4.1 (above the -5 floor, so not a hard fail) |
| G2 False alarms (`familiar`, `lookalike_heavy`) | pass | false-alarm differences -0.4 [-1.3, 0.4] and -0.8 [-1.9, 0.3] |
| G3 Memory | pass | 100 / 100 / 100 familiar, 98.6 unseen, shuffled 46.9, beats pooled [49.4, 50.6] |
| G4 Cost | pass | latency 0.82x, memory 1.00x public Sona |

**Formal verdict on the evaluator-owned sealed run: INSUFFICIENT EVIDENCE.**

**Gap in the predeclared gates.** G2 checks false alarms only on `familiar` and `lookalike_heavy`. The candidate's largest regression is on unseen conditions those gates do not cover: +19.9 points of actual false alarms on `long_sparse` and +4.1 on `dense`. The gates are kept as declared; this regression is reported here and drives the recommendation below.

### 6.5 Replication: candidate-owner-run sealed evaluation

The candidate owner ran the same unchanged suite independently at `data/generated/sona_candidate_independent_final_v1`, with the same registration.

- Its candidate freeze was recorded 10 ms before its OS-drawn seeds (415766867, 797668451, 233237749).
- All three of its replicate audits passed, and its run-level audit reports nothing changed.
- I recomputed its gates read-only without writing into that directory: `data/generated/sona_evaluation_v1/replication_codex_run_gates.json`.

**Findings:**

- **Macro recall:** -0.67 [-0.96, -0.37] versus the Sona counterpart, entirely below zero, so the predeclared verdict for that run is **FAIL**.
- **`long_sparse` false alarms:** 26.8% versus 5.1% (+21.8 [19.7, 23.8]).
- **`short` recall:** -4.6 [-5.8, -3.4].
- **`slower` recall:** -2.0 [-3.5, -0.6].
- **Memory:** 100 / 100 / 100 at familiar gaps, 98.8 unseen.

Because the candidate's owner ran this evaluator, it is reported as a replication, not as the acceptance decision. Its substance matches the evaluator-owned run.

## 7. Limitations

- All data is synthetic. Generator labels define "attack". Nothing here establishes real-world intrusion-detection accuracy or attacker intent.
- Window boundaries are supplied by the generator. Real session formation is not tested.
- Saved six-class checkpoints were not trained on these windows. Their results show transfer, not their best achievable behavior.
- Sealing prevents accidental tuning on final cases. It does not defend against a deliberately adversarial process on the same machine.
- Latency is CPU-only, on one shared workstation. Every efficiency measurement ran while another session was training (load average 2 to 5), and identical architectures differed by up to about 30%. Load is recorded and every row is flagged. Cold start is dominated by imports from a Windows-mounted virtual environment.
- The Sona counterpart is near the recall ceiling on this suite, which limits how much a candidate can show on G1. The chain-memory task is solved by both GRUs and by engineered order features. It demonstrates memory but cannot rank two memory-capable designs.
- Bootstrap intervals resample windows within each replicate. Training-seed variance appears only in the per-seed ranges.
- The predeclared G2 covers only two conditions, so it missed the candidate's long-window false-alarm regression (6.3, 6.4).
- Beta counterparts' predeclared thresholds are inflated by benign chain swaps in validation. A labeled post-hoc sensitivity table is reported, but it is not used in the gates.
- Fit mode measures the candidate's architecture and recipe on this suite. The candidate's packaged checkpoints could not be evaluated here because of their 160-flow cap.

## 8. Recommendation

**Do not adopt GRU40 as a replacement for Sona.**

**Formal outcome:**

- **Evaluator-owned sealed run:** INSUFFICIENT EVIDENCE. G1 is not met; G0, G2, G3 and G4 pass.
- **Candidate-owner replication:** FAIL under the same rules.

**Where the candidate beats Beta 1/2:** everywhere.

- About 89 to 90 points macro recall over the retrained Beta counterparts.
- About 96 points over the zero-shot saved checkpoints.
- It solves order-dependent chains that every order-invariant Beta model scores at exactly chance.

**Where it loses:**

- **Against the Sona counterpart (GRU32):** worse recall on short windows (-5.4 points), and more false alarms on longer (+19.9) and denser (+4.1) windows than in training. These have a single identified cause: the relative elapsed-units input is out of range under duration and density shift.
- **Against the Beta counterparts:** the same long-window and dense false-alarm rates (24.6% and 11.0%), where the Beta models flag almost nothing at their inflated thresholds.

**Memory:** demonstrated (100% familiar, 98.6% unseen gaps, destroyed by shuffling), but it is **not a benefit over the existing Sona architecture**, which already solves the same triples. Hand-engineered order features solve them too, at short gaps.

**Compute:** cost is not a barrier: 8,002 parameters, about 1.1x the Sona counterpart's latency, similar memory. But with no accuracy gain, the added size buys nothing measurable here.

**Suggested next steps (recommendations only, nothing was implemented):**

1. Make relative timing invariant to event count, or bound it (for example a windowed or rate-normalized elapsed feature), and test it on `long_sparse`, `dense` and `short` before any promotion.
2. Extend G2-style false-alarm gates to every shifted condition in the next protocol version.
3. Add harder memory cases on which GRU32 is not already at ceiling (longer chains, interleaved partial chains, multiple decoy bursts), so memory-capable designs can be ranked.
4. Run the separate six-category regression only after a binary candidate clears these gates.

None of these synthetic results establishes real-world intrusion-detection accuracy.
