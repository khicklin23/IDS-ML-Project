---
id: 01a1281f-b2f1-7490-8736-764d1b7022a6
slug: tasks/backend-polish
title: "Backend hardening (ongoing, post-handoff)"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
---

Ongoing backend work while a separate agent (if assigned) builds tasks/static-dashboard-ui against the report.json contract. Do not block the UI task; report.json's shape (REPORT_VERSION) should stay stable even as these improve numbers underneath it. If a change here would alter report.json's shape, bump REPORT_VERSION and flag it in tasks/static-dashboard-ui rather than silently changing the contract.

Known items, roughly in priority order:
1. ~~split_dataset has no label/scenario stratification~~ -- **FIXED, 2026-10-10.** `split_dataset` now allocates sessions per scenario family (every session is 100% one scenario by construction) instead of shuffling all sessions together, so every scenario present in the full dataset gets at least one session in every split, for every seed -- no more seed roulette. Verified: all 9 previously-bad seeds from the showcase-scale scan now pass; new regression test `tests/test_generator.py::test_split_is_stratified_by_scenario_across_many_seeds` checks this across 20 seeds on one dataset. Existing no-leakage/determinism tests still pass unchanged. **Practical effect: any seed can be used now** -- the earlier workarounds (UI default bumped to 6 episodes/family, showcase seed hand-picked to 1) are no longer load-bearing, just harmless leftovers.
2. ~~Multiclass beaconing/benign confusion~~ -- **largely resolved by scale, 2026-10-10.** At 21k rows (62-325 beaconing test examples depending on run) beaconing F1 ranged 0.57-0.81, the weakest class. At the ~50k-row validation run (item 3, 2000 beaconing test examples), beaconing F1 is **0.982** (precision 0.968, recall 0.998) -- more examples alone fixed most of the confusion. Not pursuing the inter-arrival-regularity feature idea right now since the real target scale already performs well; revisit only if a future smaller/faster demo run still shows it as a weak point.
3. ~~Scale validation~~ -- **DONE, 2026-10-10.** 200 hosts / 8hr window / 25 episodes per attack family -> 49,014 rows, built in 161.6s (~2.7 min) on this machine/filesystem. Full class coverage confirmed (split-stratification fix above). Results: binary model F1 0.991 vs rule-baseline F1 0.324 (recall 0.996 vs 0.194 -- an even starker model-vs-rule gap than the smaller runs); multiclass accuracy 0.982, macro-F1 0.984, every class F1 >= 0.94. Saved at `data/generated/showcase_50k/` (gitignored, present on disk) -- **this is now the primary example for tasks/static-dashboard-ui**, closer to the user's stated ~50k "final test run" scale and with better numbers than the smaller `data/generated/showcase/` (kept too, useful for faster dev iteration).
4. Dataset realism beyond IPs: src_ip/dst_ip were added (schema v1.1.0) for the 'what a normal IDS would see' ask. Further realism (e.g. TTL, flags, inter-arrival jitter as actual features) is open-ended future work, not required for the current handoff.

Acceptance: no regressions in the existing suite (68 tests as of this writing); any numbers used in user-facing reporting should be from a run that passed the class-coverage check in item 1 -- trivially true for any seed now that item 1 is fixed.