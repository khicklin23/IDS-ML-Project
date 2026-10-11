"""v1 -> v2 -> v3 progression (tasks/model-versions): one shared harness,
real measured differences, not just narrative claims.

v1: per-flow MLP, normal-only training.
v2: same architecture, normal+stealth mixed training (the fix validated in
    test_robustness.py).
v3: session-level GRU, normal-only training -- same recipe as v1, so any
    difference from v1 isolates the effect of architecture alone.

All three scored on session-level macro-F1 (v1/v2 rolled up via majority
vote) so the comparison is apples-to-apples.
"""

from ids_ml.models.sequence import SequenceTrainConfig
from ids_ml.models.train import TrainConfig
from ids_ml.versions import run_comparison

_TRAIN_CONFIG = TrainConfig(seed=0, max_epochs=25, patience=6)
_SEQUENCE_CONFIG = SequenceTrainConfig(seed=0, max_epochs=25, patience=6)


def test_v1_v2_v3_progression_on_shared_harness(tmp_path):
    result = run_comparison(
        tmp_path, seed=7, num_hosts=150, duration_minutes=180.0, episodes_per_family=50,
        train_config=_TRAIN_CONFIG, sequence_config=_SEQUENCE_CONFIG,
    )
    v1, v2, v3 = (result["versions"][v] for v in ("v1", "v2", "v3"))

    # All three should handle ordinary (non-stealth) traffic well -- the
    # comparison is about stealth robustness, not whether any of them work.
    assert v1["normal_session_metrics"]["macro_f1"] > 0.85, v1["normal_session_metrics"]
    assert v2["normal_session_metrics"]["macro_f1"] > 0.85, v2["normal_session_metrics"]
    assert v3["normal_session_metrics"]["macro_f1"] > 0.85, v3["normal_session_metrics"]

    v1_stealth = v1["stealth_session_metrics"]["macro_f1"]
    v2_stealth = v2["stealth_session_metrics"]["macro_f1"]
    v3_stealth = v3["stealth_session_metrics"]["macro_f1"]

    # v2's fix (training-recipe change) substantially beats v1 on stealth.
    assert v2_stealth > v1_stealth + 0.2, (v1["stealth_session_metrics"], v2["stealth_session_metrics"])
    # v3's fix (architecture change, same normal-only training as v1)
    # *also* beats v1 on stealth -- a different improvement axis producing
    # real robustness without ever training on stealth data. Smaller,
    # more training-duration-sensitive margin than v2's (observed 0.07-0.16
    # across runs) since this is a genuinely smaller effect, not noise.
    assert v3_stealth > v1_stealth + 0.05, (v1["stealth_session_metrics"], v3["stealth_session_metrics"])

    # Artifacts actually saved, not just computed in memory.
    assert (tmp_path / "v1" / "model.pt").exists()
    assert (tmp_path / "v2" / "model.pt").exists()
    assert (tmp_path / "v3_model.pt").exists()
    assert (tmp_path / "comparison.json").exists()
