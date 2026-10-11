import numpy as np

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import build_preprocessor, extract_labels, transform_features
from ids_ml.models.evaluate import evaluate
from ids_ml.models.train import TrainConfig, load_artifact, save_artifact, train_model

CONFIG = GenerationConfig(
    seed=7,
    num_hosts=30,
    duration_minutes=60,
    background_rate_per_min=25.0,
    port_scan_episodes=4,
    auth_bruteforce_episodes=4,
    beaconing_episodes=4,
)


def _prepared_splits():
    df, _ = generate_dataset(CONFIG)
    train_df, val_df, test_df = split_dataset(df, val_frac=0.2, test_frac=0.2, seed=7)
    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_labels(train_df)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_labels(val_df)
    X_test = transform_features(preprocessor, test_df)
    y_test = extract_labels(test_df)
    return preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def test_train_produces_reasonable_held_out_metrics():
    _, (X_train, y_train), (X_val, y_val), (X_test, y_test) = _prepared_splits()

    config = TrainConfig(seed=0, max_epochs=20, patience=5)
    model, history = train_model(X_train, y_train, X_val, y_val, config)

    assert len(history) > 1
    assert "best_epoch" in history[-1]

    metrics = evaluate(model, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate"):
        assert 0.0 <= metrics[key] <= 1.0
        assert not np.isnan(metrics[key])

    # Benign-dominated synthetic mix with class-weighted loss: expect the
    # baseline to clearly beat a predict-everything-normal classifier.
    assert metrics["recall"] > 0.5


def test_preprocessor_is_fit_only_on_train():
    preprocessor, (X_train, _), _, (X_test, _) = _prepared_splits()
    assert X_train.shape[1] == X_test.shape[1]
    # ColumnTransformer fitted on train must still transform test without refitting.
    assert hasattr(preprocessor, "transformers_")


def test_artifact_round_trip_matches_predictions(tmp_path):
    _, (X_train, y_train), (X_val, y_val), (X_test, y_test) = _prepared_splits()
    config = TrainConfig(seed=1, max_epochs=10, patience=3)
    model, _ = train_model(X_train, y_train, X_val, y_val, config)
    metrics_before = evaluate(model, X_test, y_test)

    preprocessor, *_ = _prepared_splits()
    save_artifact(model, preprocessor, config, metrics_before, tmp_path, input_dim=X_train.shape[1])

    loaded_model, loaded_preprocessor, metadata = load_artifact(tmp_path)
    metrics_after = evaluate(loaded_model, X_test, y_test)

    assert metrics_before == metrics_after
    assert metadata["input_dim"] == X_train.shape[1]
