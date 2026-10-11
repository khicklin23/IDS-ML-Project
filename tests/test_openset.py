"""Open-set/unknown-attack detection (tasks/model-maturity item 3).

The honest finding, from real experimentation (not assumed): max-softmax
rejection genuinely works when a held-out attack family has a distinctive
*instantaneous* per-flow signature (port_scan: tiny packets, rejected/
timeout outcomes, random high ports) -- holding it out of training and
testing against it shows real, usable detection. It essentially does NOT
work for a purely *temporal/behavioral* pattern (beaconing: the signal is
regularity across many flows over time, not any single flow's features) --
confirmed with both softmax-confidence and feature-space-distance methods
during development (see tasks/model-maturity notes). That negative result
is reported as-is, not hidden -- it's real evidence for why
tasks/temporal-modeling (session-level modeling) is worth building, not a
guess.
"""

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import build_preprocessor, extract_labels_for_classes, transform_features
from ids_ml.models.train import TrainConfig, train_multiclass_model
from ids_ml.openset import UNKNOWN_LABEL, classify_with_rejection, evaluate_open_set
from ids_ml.models.evaluate import predict_class_proba

_FAST_TRAIN = TrainConfig(seed=0, max_epochs=35, patience=8)


def _train_restricted(train_config: GenerationConfig, known_classes: list[str]):
    df, _ = generate_dataset(train_config)
    train_df, val_df, _ = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=train_config.seed)
    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_labels_for_classes(train_df, known_classes)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_labels_for_classes(val_df, known_classes)
    model, _ = train_multiclass_model(X_train, y_train, X_val, y_val, _FAST_TRAIN, len(known_classes))
    return model, preprocessor


def test_classify_with_rejection_basic_behavior():
    import numpy as np
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    labels = classify_with_rejection(probs, ["a", "b", "c"], threshold=0.6)
    assert labels == ["a", UNKNOWN_LABEL]


def test_open_set_detects_a_held_out_class_with_a_distinctive_instantaneous_signature():
    """port_scan held entirely out of training: tiny/rejected/timeout flows
    on random ports look nothing like any trained class's typical flow, so
    confidence-based rejection should catch a meaningful fraction of it.
    """
    known_classes = ["benign", "auth_bruteforce", "beaconing", "service_flood", "outbound_transfer"]
    train_config = GenerationConfig(
        seed=50, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
        port_scan_episodes=0, auth_bruteforce_episodes=10, beaconing_episodes=10,
        service_flood_episodes=10, outbound_transfer_episodes=10,
    )
    open_test_config = GenerationConfig(
        seed=60, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
        port_scan_episodes=15, auth_bruteforce_episodes=10, beaconing_episodes=10,
        service_flood_episodes=10, outbound_transfer_episodes=10,
    )
    model, preprocessor = _train_restricted(train_config, known_classes)
    open_test_df, _ = generate_dataset(open_test_config)

    result = evaluate_open_set(model, preprocessor, open_test_df, known_classes, threshold=0.9)
    assert result["novel_scenarios"] == ["port_scan"]
    assert result["novel_detection_rate"] > 0.25, result
    assert result["known_accuracy"] > 0.7, result
    assert result["known_false_rejection_rate"] < 0.2, result

    # Raising the threshold trades more novel detection for more false
    # rejection of known classes -- a real operating-point tradeoff, not a
    # free improvement. Confirm both directions move as expected.
    stricter = evaluate_open_set(model, preprocessor, open_test_df, known_classes, threshold=0.99)
    assert stricter["novel_detection_rate"] > result["novel_detection_rate"]
    assert stricter["known_false_rejection_rate"] > result["known_false_rejection_rate"]


def test_open_set_fails_on_a_purely_temporal_pattern_honestly():
    """beaconing held entirely out of training: its per-flow features
    (small https flows, established outcome) look like ordinary benign
    traffic -- the actual signal (regular timing across many flows) isn't
    visible to a single-flow classifier no matter the confidence
    threshold. This asserts the known limitation stays a known limitation
    (i.e. catches a regression if something makes it look fixed by
    accident) and documents why in the module/task docs rather than the UI.
    """
    known_classes = ["benign", "port_scan", "auth_bruteforce", "service_flood", "outbound_transfer"]
    train_config = GenerationConfig(
        seed=50, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
        port_scan_episodes=10, auth_bruteforce_episodes=10, beaconing_episodes=0,
        service_flood_episodes=10, outbound_transfer_episodes=10,
    )
    open_test_config = GenerationConfig(
        seed=60, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
        port_scan_episodes=10, auth_bruteforce_episodes=10, beaconing_episodes=15,
        service_flood_episodes=10, outbound_transfer_episodes=10,
    )
    model, preprocessor = _train_restricted(train_config, known_classes)
    open_test_df, _ = generate_dataset(open_test_config)

    result = evaluate_open_set(model, preprocessor, open_test_df, known_classes, threshold=0.95)
    assert result["novel_scenarios"] == ["beaconing"]
    assert result["novel_detection_rate"] < 0.05, result
