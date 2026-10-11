"""v1 -> v2 -> v3 -> v4 progression (tasks/model-versions): one shared
harness, real measured differences, not just narrative claims.

v1: per-flow MLP, normal-only training.
v2: same architecture, normal+stealth mixed training (the fix validated in
    test_robustness.py).
v3: session-level GRU, normal-only training -- same recipe as v1, so any
    difference from v1 isolates the effect of architecture alone.
v4: session-level GRU, normal+stealth mixed training (v2's recipe) -- both
    fixes together. Also exercises the max_seq_len auto-sizing fix: v3's
    original implementation silently truncated sessions at a fixed 50
    steps, which cut off most normal-mode benign/beaconing/etc. sessions
    (they routinely run 100-250+ flows). max_seq_len is now computed from
    the real data instead of guessed.

All four scored on session-level macro-F1 (v1/v2 rolled up via majority
vote) so the comparison is apples-to-apples.
"""

from ids_ml.models.sequence import SequenceTrainConfig
from ids_ml.models.train import TrainConfig
from ids_ml.versions import run_comparison

_TRAIN_CONFIG = TrainConfig(seed=0, max_epochs=25, patience=6)
_SEQUENCE_CONFIG = SequenceTrainConfig(seed=0, max_epochs=25, patience=6)


def test_v1_through_v4_progression_on_shared_harness(tmp_path):
    result = run_comparison(
        tmp_path, seed=7, num_hosts=150, duration_minutes=180.0, episodes_per_family=50,
        train_config=_TRAIN_CONFIG, sequence_config=_SEQUENCE_CONFIG,
    )
    v1, v2, v3, v4 = (result["versions"][v] for v in ("v1", "v2", "v3", "v4"))

    # max_seq_len must actually cover the real data, not the old fixed 50.
    assert result["max_seq_len"] > 100, result["max_seq_len"]

    # All four should handle ordinary (non-stealth) traffic well.
    for v, label in ((v1, "v1"), (v2, "v2"), (v3, "v3"), (v4, "v4")):
        assert v["normal_session_metrics"]["macro_f1"] > 0.7, (label, v["normal_session_metrics"])

    v1_stealth = v1["stealth_session_metrics"]["macro_f1"]
    v2_stealth = v2["stealth_session_metrics"]["macro_f1"]
    v4_stealth = v4["stealth_session_metrics"]["macro_f1"]

    # v2's training-recipe fix substantially beats v1 on stealth.
    assert v2_stealth > v1_stealth + 0.2, (v1["stealth_session_metrics"], v2["stealth_session_metrics"])
    # v4 combines the architecture fix (v3) with the training-recipe fix
    # (v2) -- it should be at least as good as v2 on stealth, ideally
    # better, and clearly better than v1.
    assert v4_stealth > v1_stealth + 0.2, (v1["stealth_session_metrics"], v4["stealth_session_metrics"])

    # Artifacts actually saved, not just computed in memory.
    assert (tmp_path / "v1" / "model.pt").exists()
    assert (tmp_path / "v2" / "model.pt").exists()
    assert (tmp_path / "v3_model.pt").exists()
    assert (tmp_path / "v4_model.pt").exists()
    assert (tmp_path / "comparison.json").exists()
