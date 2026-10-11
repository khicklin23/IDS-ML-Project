---
id: 01a12b39-c00c-70b3-a495-ab16642e2e14
slug: tasks/model-versions
title: "Model version registry: v1 -> v2 -> v3 -> v4 progression"
type: task
status: completed
priority: high
agent_id: agent-2-env-setup
---

User-directed, 2026-10-11: version the attack-type classifier explicitly (v1/v2/v3) rather than just overwriting it, so improvement over time is measurable on a shared eval harness, not just claimed. Three versions planned for this round:

v1: per-flow MLP baseline, trained on one normal (non-stealth) generation run. This is what tasks/attack-type-classifier and tasks/model-maturity already built and validated -- repackaged as the v1 artifact under the registry, not rebuilt from scratch.
v2: same per-flow MLP architecture, but the training recipe validated in tests/test_robustness.py as standard practice: mixed normal+stealth training data. Already proven in testing to recover most stealth-robustness lost by v1 without sacrificing normal-case performance. Repackaged as v2, not new modeling work.
v3: session-level sequence model (small GRU) operating on a session's ordered sequence of flows instead of one flow at a time. New work -- directly motivated by tasks/model-maturity's open-set finding that beaconing's signal (timing regularity across many flows) is structurally invisible to any per-flow feature set, however augmented. Compared against v1/v2 on the same axis: roll per-flow models' predictions up to session-level (majority vote) so all three are judged on session-level accuracy/macro-F1, not an apples-to-oranges comparison.

Deliverable: a shared comparison harness/report showing all three versions' metrics side by side (session-level accuracy, macro-F1, per-class, and stealth-robustness where applicable) so the progression is visible as data, not prose. Keep each version's artifact on disk (don't overwrite) so the comparison is reproducible later. No UI work; report.json contract stays as-is unless a real need to extend it emerges (flag in tasks/static-dashboard-ui if so).

### Built, 2026-10-11 (agent-2-env-setup)

Files: `src/ids_ml/models/sessions.py` (`build_session_sequences` -- turns a flow DataFrame into padded per-session sequences using the *same* preprocessed features v1/v2 use, so only architecture differs), `src/ids_ml/models/sequence.py` (`SequenceClassifier`: one `nn.GRU` + linear head over packed variable-length sequences; `train_sequence_model` mirrors `train_multiclass_model`'s shape -- seeded, validation-based early stopping, history log), `src/ids_ml/versions.py` (`run_comparison`: generates one normal + one stealth dataset, trains all three versions, rolls v1/v2's per-flow predictions up to session-level via majority vote for a fair comparison against v3's native session-level output, saves every artifact + `comparison.json`). Refactored `models/evaluate.py` to expose `multiclass_metrics_from_predictions`/`multiclass_metrics_from_probs` as shared scoring so the per-flow model, the session rollup, and the sequence model are all scored identically.

**Scale note**: session-level sample size is sessions, not flows -- the configs used everywhere else in this project (tens of episodes) give only ~1 session per class at the test split, useless for evaluation. Tuned to `num_hosts=150, episodes_per_family=50` (duration 180min): ~360 total sessions, ~34 train / 8 test per attack class, ~78 train / 16 test benign -- small by ML standards but enough for a real, if noisy, comparison. `tests/test_generator.py`'s existing per-row-count configs are unaffected; this is purely a `versions.py`-level tuning choice.

**Result**, confirmed across two runs (`tests/test_model_versions.py`'s reduced-epoch config, and a full-default-epoch run saved at `data/generated/model_versions/comparison.json`): all three handle normal (non-stealth) traffic perfectly at this scale (session macro-F1 1.0 for all three -- 56 normal test sessions). On stealth-evasion traffic neither v1 nor v3 ever saw in training:

| | v1 (baseline) | v2 (mixed training) | v3 (session GRU) |
|---|---|---|---|
| reduced-epoch run | 0.55-0.57 | 1.0 | 0.64-0.71 |
| full-epoch run | 0.598 | 1.0 | 0.619 |

**v2's fix is large and consistent**: ~1.0 both times -- the training-recipe fix from tasks/model-maturity, now confirmed at session level too. **v3's fix is real but smaller and noisier**: it beat v1 in every run (never worse), but the margin ranged +0.02 to +0.16 across runs/epoch budgets rather than a single clean number. Being honest about why: the stealth test split is only 56 sessions (~9 per attack class) -- small enough that the exact margin is sensitive to training duration and random seed, even though the *direction* (v3 > v1) held every time it was checked. Reported as a real-but-modest, somewhat noisy effect, not a single cherry-picked number. The underlying mechanism (seeing a session's whole sequence gives some inherent robustness to pacing changes, for free, without ever training on stealth data) is the actual finding -- the precise percentage is secondary and would benefit from a larger stealth-test sample if this becomes a headline number anywhere.

Two different, real improvement axes: v2 improves via better training data, v3 improves via better architecture -- both beat v1, independently, neither a replacement for the other.

## v4, 2026-10-11 (agent-2-env-setup, user-directed follow-up)

User looked at the v3 numbers above in the UI and asked, specifically: why is v3 worse than v2 on paper, is beaconing still broken, do we need more data, and "how can we improve v3 to make it the ultra model." Investigated before touching anything, per usual practice here -- the answer wasn't "throw more data at it" alone, it was a real bug plus an unfair comparison:

**Bug found and fixed: `max_seq_len` was a fixed guess (50), not computed from real data.** Checked actual session lengths at this project's scale: normal-mode benign averages ~97-110 flows, beaconing ~139 (up to 239), service_flood ~180 (up to 247), port_scan ~127. At `max_seq_len=50`, v3 was training and evaluating on only the *first* fraction of exactly the sessions hardest to tell apart (benign vs. beaconing) -- never seeing most of the actual periodic pattern. This wasn't a deliberate experimental control, just an unchecked guess; fixed by computing `max_seq_len` from the real data (`versions.py::_max_session_length`, +10% headroom) instead of hardcoding it. This fix applies to v3 too, not just new work -- v3's number in this doc is now the corrected one, not grandfathered under the bug.

**Unfair comparison**: v3 was deliberately trained normal-only (same recipe as v1) to isolate "does architecture alone help," while v2 got to train on stealth examples too. v3 losing to v2 on stealth was never surprising on its own terms -- it just hadn't been given v2's advantage yet.

**v4 = v3's architecture + v2's training recipe + a bigger dataset** (user explicitly asked for all three: mixed stealth data, "fine-tune everything," and "up the dataset size"). Scaled from `num_hosts=150/episodes=50` to `num_hosts=200/episodes=100` (~2x sessions per class: 70 train/15 test per attack family, 104 train/22 test benign, vs. ~34/8 and ~78/16 before) for a more stable read, and reran **all four** versions on this same bigger dataset so the v4-vs-others comparison isn't confounded by dataset size.

### Result (`data/generated/model_versions/comparison.json`, full run)

| | v1 | v2 | v3 (bug fixed) | v4 |
|---|---|---|---|---|
| Normal session macro-F1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Stealth session macro-F1 | 0.562 | **1.0** | 0.599 | **1.0** |

v4 matches v2's perfect stealth score -- and checked the **per-class** breakdown, not just the aggregate, specifically because the user flagged beaconing: v4 gets precision=recall=f1=1.0 on *every single class* on both normal and stealth test sets, beaconing included. v1 and v3 (bug-fixed but still normal-only-trained) both show beaconing and auth_bruteforce at precision=recall=0 on stealth -- confidently misclassified as benign, confirming the earlier finding that an architecture fix alone doesn't substitute for actually training on the evasive examples; it needs both.

**Honest scope note**: these are near-perfect scores on *held-out sessions from the same scenario-generation distribution* (same config, different random episodes) -- not the harder "genuinely different distribution shift" bar from `tests/test_robustness.py` (different seed/host-count/duration/traffic-mix entirely), which is a separate, already-passing test and wasn't rerun as part of this change. Worth remembering before calling v4 bulletproof against *every* possible variation, not just the stealth-pacing one tested here.

`tests/test_model_versions.py` updated to cover all four versions and assert `max_seq_len` is actually computed from real data (>100, not the old fixed 50). 85/85 project tests pass. `versions.py`'s `VERSIONS_SCHEMA` bumped 1.0.0 -> 2.0.0 (new version + the bug fix changes v3's numbers, not just adds v4). Artifacts: `v1/`, `v2/` (full bundles), `v3_model.pt`+`v3_metadata.json`, `v4_model.pt`+`v4_metadata.json`, `comparison.json`, replacing the old (bug-affected) `data/generated/model_versions/` directory rather than living alongside it as a confusing second copy.

**Not done / honest limits**: no architecture tuning beyond the data fixes (hidden_dim, pooling strategy, bidirectional GRU, etc. all untouched) -- wasn't needed once the real bugs were fixed, so didn't add complexity without evidence it was required. Stealth test set is still only 97 sessions; perfect scores there are a real, verified result at this scale, not a guarantee the next distribution flung at it also hits 100%.

### UI sync, 2026-10-11 (same day, following)

The user was reading the static dashboard's version comparison page while asking why v3 lagged v2 ("61.9%", "40 out of 56") -- those exact numbers matched the *old, bug-affected* `comparison.json` that `ui/versions.json` still had a stale copy of, built by agent-1-version-ui (`tasks/version-overview-ui`, completed, v1-v3 only at the time) before this task's bug fix and v4 existed. The UI itself wasn't broken; it was correctly rendering stale data plus logic hardcoded to exactly three versions (`latestVersion` literally `'v3'`, a fixed "v2 has the strongest result" narrative sentence, dropdown capped at v3).

Fixed within this task rather than filing a separate one, since it's a direct, necessary consequence of the backend fix actually reaching the user:
- Copied the corrected `data/generated/model_versions/comparison.json` -> `ui/versions.json`.
- `ui/versions-data.mjs`: added a `v4` entry to `versionNotes`; relaxed `validateVersions`'s schema check from an exact `'1.0.0'` string match to "must be a string" (forward-compatible with the `VERSIONS_SCHEMA` bump to `2.0.0`).
- `ui/versions.js`: replaced the hardcoded `'v3'` with `latestVersion` computed from `Object.keys(versionNotes)`; rewrote the findings sentence to compute which version(s) actually have the best stealth score instead of assuming v2, so it won't silently go stale again if a v5 ties or wins.
- `ui/index.html`: added the `v4` option to `#version-select`, moved `selected` to it.
- `ui/checks/versions.test.mjs`: updated every hardcoded expected number to the real new ones (v1 62/97 56.2%, v2 97/97 100.0%, v3 67/97 59.9%, v4 97/97 100.0%) and fixed the schema-mutation test (`d.versions_schema=9`, a number, now that the check is type-based rather than exact-match).
- Verified: `npm run check` all 26 tests pass; local `http.server` sanity check returned 200 for `index.html`, `versions.js`, `versions-data.mjs`, `versions.json`.

Did not touch anything outside the version-comparison section -- same coordination-boundary discipline as the rest of `tasks/version-overview-ui`'s scope. No browser visual check (no browser tool in this environment); verification is the test suite plus the HTTP 200 check, same substitute used throughout this project's UI work.

### Second real bug found, 2026-10-11 (while building the v4 report export): session_id collisions between normal and stealth runs

User flagged a separate gap -- `tasks/dashboard-data-refresh` noted the Overview dashboard couldn't show v4 detections because no alerts/raw-data export existed for it, only `versions.py`'s comparison numbers. Building that export (`ids_ml.report.build_session_report`, below) required combining a normal-pace generation run with a stealth-pace one the same way `versions.py` already does for `mixed_train`/`mixed_val` -- and checking that combination carefully (since the new function evaluates on a *combined* test set too, which `versions.py` never does) surfaced a real, previously-undetected bug in that exact combination step.

**The bug**: `session_id` naming doesn't depend on the generation seed -- benign sessions are named directly from `host_id`, attack sessions from `scenario+episode-index`. Two separately-seeded runs (normal at `seed`, stealth at `seed+1000`) with the same host count and episode counts therefore produce **the same session_id strings**. Verified directly: at production scale (200 hosts, 100 episodes/family), 607 of 648 session_ids (94%) collided between a normal run and a stealth run. Every prior place that combined the two with a plain `pd.concat` (`versions.py`'s `mixed_train`/`mixed_val`, used by v2 and v4) was silently merging two *different* episodes' flows into one artificial session under any `groupby("session_id")` -- invisible to v2 (per-flow, never groups by session) but real for v4 (session-level GRU, which groups by session_id to build training sequences).

**The fix**: `ids_ml.data.generator.combine_runs(*dfs)` -- concatenates like `pd.concat` but first namespaces `session_id` as `f"{run_id}::{session_id}"` per source run, guaranteeing uniqueness. Verified empirically eliminates the collision (1297 unique sessions from 1297 total, matching normal+stealth session counts exactly). Applied everywhere two runs get combined: `versions.py`'s `mixed_train`/`mixed_val`, and `report.py`'s `full_df`/`mixed_train`/`mixed_val`/`test_df`.

**Did it change the headline numbers?** Reran all four versions on the same real-scale config (200 hosts, 100 episodes/family) after the fix:

| | v1 | v2 | v3 | v4 |
| --- | --- | --- | --- | --- |
| Normal session macro-F1 | 1.0 | 1.0 | 1.0 | **0.9907** (was 1.0) |
| Stealth session macro-F1 | 0.562 | 1.0 | 0.599 | 1.0 (unchanged) |

Only v4 moved, and only on the normal-pace test (stealth result unchanged). v1/v3 are normal-only and never touch `mixed_train`; v2 uses `mixed_train` but is per-flow and never groups by session, so the fix is a no-op for it. v4 is the only path that both combines runs *and* groups by session_id, so it's the only one the bug could have touched -- consistent with the measured outcome, not a surprise. The small drop (1.0 -> 0.9907) is the *more* trustworthy number: a flat 1.0 across every single class was already slightly suspicious, and removing a real data-corruption bug landing just shy of perfect is the expected direction, not a regression to explain away. README and this doc updated to the corrected number; stealth's perfect score (the headline result the user cares about -- beaconing, specifically) is unaffected.

### `ids_ml.report.build_session_report`: the v4 backend export `tasks/dashboard-data-refresh` was waiting on

Added `build_session_report()` to `src/ids_ml/report.py`, parallel to the existing `build_report()` (per-flow MLP) but using the session GRU (v4 recipe: architecture + mixed training) as the detector. Produces the exact same `report.json` shape/contract -- validated against the UI's own `ui/data.mjs::validateReport()`, which accepts it unmodified, no UI changes needed. The one real design choice, documented in the report's own `detector` field: a session model predicts one label per *session*, not per flow, so every flow in a session inherits that session's `predicted_scenario`/confidence, and `model_score = 1 - P(benign)` from that same softmax stands in for the separate binary head v1/v2 trained directly -- `binary_model` and `multiclass_model` in the output are two views of the same trained GRU, not two separate models, and the report says so.

`tests/test_report.py::test_session_report_matches_the_same_documented_shape` added (contract test, fast-training config). All 86 project tests pass. Full-scale bundle written to `data/generated/showcase_v4/` (seed 7, same 200-host/180-min/100-episodes-per-family config as the comparison above, post-fix) -- `report.json`, `dataset/flows.parquet`, `artifact_sequence/` (model + preprocessor + metadata), `alerts.db`/`alerts.csv`. Not yet wired up as the UI's default bundled report -- that's a UI-side decision (`tasks/dashboard-data-refresh`'s agent or whoever picks this up), this task only had to prove the backend export exists and is contract-valid.