"""Confidence calibration (tasks/model-maturity item 5): is "90% confident"
actually right ~90% of the time? Reported for transparency; not asserted
to be good, since it isn't -- see the real-model test below and
tasks/model-maturity's notes on why (same overconfidence that broke naive
open-set rejection in ids_ml.openset).
"""

import numpy as np

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import SCENARIO_CLASSES, build_preprocessor, extract_scenario_labels, transform_features
from ids_ml.models.evaluate import expected_calibration_error, predict_class_proba
from ids_ml.models.train import TrainConfig, train_multiclass_model


def test_perfectly_calibrated_toy_example_has_near_zero_ece():
    # 100 predictions at 0.7 confidence, 70 correct -- confidence matches accuracy exactly.
    rng = np.random.default_rng(0)
    probs = np.zeros((100, 2))
    probs[:, 0] = 0.7
    probs[:, 1] = 0.3
    y = np.zeros(100, dtype=np.int64)
    wrong_idx = rng.choice(100, size=30, replace=False)
    y[wrong_idx] = 1

    result = expected_calibration_error(probs, y, n_bins=10)
    assert result["ece"] < 0.02, result


def test_overconfident_toy_example_has_high_ece():
    # Always 99% confident, but only actually right 80% of the time.
    rng = np.random.default_rng(0)
    probs = np.full((200, 2), 0.005)
    probs[:, 0] = 0.99
    y = np.zeros(200, dtype=np.int64)
    wrong_idx = rng.choice(200, size=40, replace=False)
    y[wrong_idx] = 1

    result = expected_calibration_error(probs, y, n_bins=10)
    assert result["ece"] > 0.15, result


def test_real_multiclass_model_calibration_is_computed_and_reported():
    """Not asserting good calibration -- asserting it's measured honestly.
    The real finding (consistent with tasks/model-maturity's open-set
    results): this model is meaningfully overconfident.
    """
    config = GenerationConfig(
        seed=99, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
        port_scan_episodes=6, auth_bruteforce_episodes=6, beaconing_episodes=6,
        service_flood_episodes=6, outbound_transfer_episodes=6,
    )
    df, _ = generate_dataset(config)
    train_df, val_df, test_df = split_dataset(df, val_frac=0.2, test_frac=0.2, seed=99)

    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_scenario_labels(train_df)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_scenario_labels(val_df)
    X_test = transform_features(preprocessor, test_df)
    y_test = extract_scenario_labels(test_df)

    model, _ = train_multiclass_model(
        X_train, y_train, X_val, y_val, TrainConfig(seed=0, max_epochs=20, patience=5), len(SCENARIO_CLASSES)
    )
    probs = predict_class_proba(model, X_test)
    result = expected_calibration_error(probs, y_test, n_bins=10)

    assert 0.0 <= result["ece"] <= 1.0
    assert sum(b["count"] for b in result["bins"]) == result["n"]
    # The honest finding: this model's confidence is not trustworthy at
    # face value -- it clusters near 100% confidence regardless of whether
    # it's actually right that often. Document this, don't paper over it.
    top_bin = max(result["bins"], key=lambda b: b["confidence"])
    assert top_bin["confidence"] > 0.95
    assert top_bin["accuracy"] < top_bin["confidence"]
