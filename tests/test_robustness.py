"""Robustness: generalization to a shifted distribution, and the honest
stealth-mode vulnerability + its fix via mixed training.

This is the actual point of tasks/model-maturity item 2: prove the model
is learning real signal (survives ordinary distribution shift) while also
being honest about where it's weak (stealth/evasion variants it never saw)
and showing that weakness is fixable (mixed training), not papering over it.
"""

import pandas as pd

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import SCENARIO_CLASSES, build_preprocessor, extract_scenario_labels, transform_features
from ids_ml.models.train import TrainConfig, train_multiclass_model
from ids_ml.robustness import evaluate_multiclass_on_dataframe, evaluate_multiclass_on_shifted_distribution

_CLASS_NAMES = list(SCENARIO_CLASSES)
_FAST_TRAIN = TrainConfig(seed=0, max_epochs=35, patience=8)

_NORMAL_CONFIG = GenerationConfig(
    seed=10, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
    port_scan_episodes=10, auth_bruteforce_episodes=10, beaconing_episodes=10,
    service_flood_episodes=10, outbound_transfer_episodes=10, stealth_mode=False,
)
_STEALTH_CONFIG = GenerationConfig(
    seed=20, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
    port_scan_episodes=10, auth_bruteforce_episodes=10, beaconing_episodes=10,
    service_flood_episodes=10, outbound_transfer_episodes=10, stealth_mode=True,
)
# A deliberately differently-shaped config: different seed, host count,
# traffic rate, and episode mix -- ordinary distribution shift, not evasion.
_SHIFTED_CONFIG = GenerationConfig(
    seed=30, num_hosts=55, duration_minutes=70, background_rate_per_min=80.0,
    port_scan_episodes=7, auth_bruteforce_episodes=12, beaconing_episodes=8,
    service_flood_episodes=9, outbound_transfer_episodes=11, stealth_mode=False,
)


def _train_multiclass(train_df, val_df):
    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_scenario_labels(train_df)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_scenario_labels(val_df)
    model, _ = train_multiclass_model(X_train, y_train, X_val, y_val, _FAST_TRAIN, len(_CLASS_NAMES))
    return model, preprocessor


def test_model_generalizes_to_an_ordinary_distribution_shift():
    """Different seed, host count, traffic rate, episode mix -- not just a
    held-out split of the same generation run. If this fails, the model
    learned generator quirks, not attack signatures.
    """
    normal_df, _ = generate_dataset(_NORMAL_CONFIG)
    train_df, val_df, _ = split_dataset(normal_df, val_frac=0.15, test_frac=0.15, seed=_NORMAL_CONFIG.seed)
    model, preprocessor = _train_multiclass(train_df, val_df)

    result = evaluate_multiclass_on_shifted_distribution(model, preprocessor, _CLASS_NAMES, _SHIFTED_CONFIG)
    assert result["metrics"]["macro_f1"] > 0.7, result["metrics"]


def test_stealth_variants_expose_a_real_weakness_that_mixed_training_fixes():
    """The full, honest robustness story (empirically derived, not assumed):

    1. Baseline (normal-only training, with the prior_gap_s inter-arrival
       feature -- see schema v1.2.0): great on normal traffic, but
       measurably worse on stealth variants it never saw. prior_gap_s is
       *exactly* what an adaptive attacker would be changing (that's the
       point of pacing an attack slower), so a feature that helps a lot on
       the training distribution can make the model *more* brittle to
       evasion, not less -- confirmed by comparison to the pre-feature
       baseline during development (see tasks/model-maturity): stealth
       macro_f1 was *worse* with the feature than without it, on
       normal-only training.
    2. Mixing stealth examples into training recovers robustness (the model
       learns the feature's informativeness is context-dependent) while
       keeping almost all of the normal-case precision gain the feature
       provides. This is the actual fix -- not the feature alone, not more
       data alone, but both together.
    """
    normal_df, _ = generate_dataset(_NORMAL_CONFIG)
    normal_train, normal_val, normal_test = split_dataset(
        normal_df, val_frac=0.15, test_frac=0.15, seed=_NORMAL_CONFIG.seed
    )

    stealth_df, _ = generate_dataset(_STEALTH_CONFIG)
    stealth_train, stealth_val, stealth_test = split_dataset(
        stealth_df, val_frac=0.15, test_frac=0.15, seed=_STEALTH_CONFIG.seed
    )

    # --- baseline: trained on normal attacks only ---
    baseline_model, baseline_preprocessor = _train_multiclass(normal_train, normal_val)
    baseline_on_normal = evaluate_multiclass_on_dataframe(baseline_model, baseline_preprocessor, _CLASS_NAMES, normal_test)
    baseline_on_stealth = evaluate_multiclass_on_dataframe(baseline_model, baseline_preprocessor, _CLASS_NAMES, stealth_test)

    assert baseline_on_normal["metrics"]["macro_f1"] > 0.9, baseline_on_normal["metrics"]
    # The actual finding: naive-trained model is measurably worse against
    # stealth variants than against the normal attacks it was trained on.
    assert baseline_on_stealth["metrics"]["macro_f1"] < baseline_on_normal["metrics"]["macro_f1"] - 0.25, (
        baseline_on_normal["metrics"], baseline_on_stealth["metrics"],
    )

    # --- fix: mix stealth examples into training ---
    mixed_train = pd.concat([normal_train, stealth_train], ignore_index=True)
    mixed_val = pd.concat([normal_val, stealth_val], ignore_index=True)
    robust_model, robust_preprocessor = _train_multiclass(mixed_train, mixed_val)

    robust_on_stealth = evaluate_multiclass_on_dataframe(robust_model, robust_preprocessor, _CLASS_NAMES, stealth_test)
    robust_on_normal = evaluate_multiclass_on_dataframe(robust_model, robust_preprocessor, _CLASS_NAMES, normal_test)

    # The fix: substantially better on stealth than the naive baseline was.
    assert robust_on_stealth["metrics"]["macro_f1"] > baseline_on_stealth["metrics"]["macro_f1"] + 0.15, (
        baseline_on_stealth["metrics"], robust_on_stealth["metrics"],
    )
    # Without regressing badly on ordinary (non-stealth) traffic.
    assert robust_on_normal["metrics"]["macro_f1"] > 0.85, robust_on_normal["metrics"]
