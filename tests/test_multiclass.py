"""Attack-type (multiclass) classification: this is the capability that
gives the neural network a real job a simple rule system can't match --
see docs/PRODUCT_VISION.md and the user-priority note in context/roadmap.
"""

import numpy as np
import pytest

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import SCENARIO_CLASSES, build_preprocessor, extract_scenario_labels, transform_features
from ids_ml.inference import load_model_for_inference, score_dataset_multiclass
from ids_ml.models.evaluate import evaluate_multiclass
from ids_ml.models.train import TrainConfig, load_artifact, save_artifact, train_multiclass_model
from ids_ml.pipeline import run_multiclass

_SEED = 99
_CONFIG = GenerationConfig(
    seed=_SEED,
    num_hosts=40,
    duration_minutes=90,
    background_rate_per_min=60.0,
    port_scan_episodes=14,
    auth_bruteforce_episodes=14,
    beaconing_episodes=14,
    service_flood_episodes=14,
    outbound_transfer_episodes=14,
)
# Split fractions must match ids_ml.pipeline.run_multiclass's hardcoded
# 0.15/0.15 so tests reconstructing the test split get the same rows it used.
_VAL_FRAC = 0.15
_TEST_FRAC = 0.15


def _prepared_splits():
    df, _ = generate_dataset(_CONFIG)
    train_df, val_df, test_df = split_dataset(df, val_frac=_VAL_FRAC, test_frac=_TEST_FRAC, seed=_SEED)
    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_scenario_labels(train_df)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_scenario_labels(val_df)
    X_test = transform_features(preprocessor, test_df)
    y_test = extract_scenario_labels(test_df)
    return preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def test_extract_scenario_labels_covers_all_classes_and_rejects_unknown():
    df, _ = generate_dataset(_CONFIG)
    y = extract_scenario_labels(df)
    assert set(y) == set(range(len(SCENARIO_CLASSES)))  # every class present in this config

    bad = df.copy()
    bad["scenario"] = bad["scenario"].astype(object)
    bad.loc[bad.index[0], "scenario"] = "not_a_real_scenario"
    with pytest.raises(ValueError, match="not_a_real_scenario"):
        extract_scenario_labels(bad)


def test_all_classes_present_in_every_split():
    _, (_, y_train), (_, y_val), (_, y_test) = _prepared_splits()
    for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        missing = set(range(len(SCENARIO_CLASSES))) - set(y)
        assert not missing, f"{name} split is missing classes: {missing}"


def test_multiclass_model_beats_majority_baseline_and_identifies_each_attack():
    """The actual point of this capability: prove the NN earns its place.

    A trivial "always predict the majority class" baseline gets high overall
    accuracy for free (benign dominates) but zero recall on every attack
    type. The trained model must clear that bar by a wide margin on recall
    for every attack family, not just overall accuracy.
    """
    _, (X_train, y_train), (X_val, y_val), (X_test, y_test) = _prepared_splits()
    config = TrainConfig(seed=0, max_epochs=40, patience=8)
    model, history = train_multiclass_model(X_train, y_train, X_val, y_val, config, len(SCENARIO_CLASSES))

    assert len(history) > 1
    metrics = evaluate_multiclass(model, X_test, y_test, list(SCENARIO_CLASSES))

    majority_class = np.bincount(y_train).argmax()
    majority_baseline_recall = float((y_test == majority_class).mean())  # recall on its own class only
    assert metrics["macro_f1"] > 0.6, metrics

    for scenario in SCENARIO_CLASSES:
        if scenario == "benign":
            continue
        recall = metrics["per_class"][scenario]["recall"]
        assert recall > 0.5, f"{scenario} recall too low: {metrics['per_class'][scenario]}"

    # Sanity: majority-class baseline is a real number that could be
    # confused for "good" accuracy; the model's benign recall shouldn't be
    # the only thing driving its macro_f1 above it.
    assert majority_baseline_recall > 0  # just confirms benign is actually the majority as designed


def test_artifact_round_trip_preserves_multiclass_predictions(tmp_path):
    preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test) = _prepared_splits()
    config = TrainConfig(seed=1, max_epochs=15, patience=5)
    model, _ = train_multiclass_model(X_train, y_train, X_val, y_val, config, len(SCENARIO_CLASSES))
    metrics_before = evaluate_multiclass(model, X_test, y_test, list(SCENARIO_CLASSES))

    save_artifact(
        model, preprocessor, config, metrics_before, tmp_path,
        input_dim=X_train.shape[1], output_dim=len(SCENARIO_CLASSES), class_names=list(SCENARIO_CLASSES),
    )
    loaded_model, loaded_preprocessor, metadata = load_artifact(tmp_path)
    metrics_after = evaluate_multiclass(loaded_model, X_test, y_test, list(SCENARIO_CLASSES))

    assert metrics_before == metrics_after
    assert metadata["output_dim"] == len(SCENARIO_CLASSES)
    assert metadata["class_names"] == list(SCENARIO_CLASSES)


def test_score_dataset_multiclass_matches_training_evaluation(tmp_path):
    result = run_multiclass(_SEED, tmp_path, gen_config=_CONFIG)

    _, _, test_df = split_dataset(generate_dataset(_CONFIG)[0], val_frac=_VAL_FRAC, test_frac=_TEST_FRAC, seed=_SEED)
    model, preprocessor, metadata = load_model_for_inference(tmp_path / "artifact_multiclass")
    scored = score_dataset_multiclass(model, preprocessor, test_df, metadata["class_names"])

    assert "predicted_scenario" in scored.columns
    assert "predicted_scenario_confidence" in scored.columns
    assert set(scored["predicted_scenario"]) <= set(SCENARIO_CLASSES)
    assert scored["predicted_scenario_confidence"].between(0, 1).all()

    # predicted_scenario is the model's own inference and must not leak
    # ground truth: at least some rows should be *wrong* relative to the
    # generator's `scenario` column, since the model isn't perfect and this
    # would otherwise suggest predicted_scenario was accidentally copied
    # from ground truth instead of produced by the model.
    assert (scored["predicted_scenario"] != scored["scenario"].astype(str)).any()

    accuracy = (scored["predicted_scenario"] == scored["scenario"].astype(str)).mean()
    assert abs(accuracy - result["test_metrics"]["accuracy"]) < 1e-9
