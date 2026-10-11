from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import extract_labels
from ids_ml.inference import load_model_for_inference, score_dataset
from ids_ml.models.evaluate import evaluate_predictions
from ids_ml.pipeline import run

_NUM_HOSTS = 30
_DURATION_MINUTES = 60


def test_score_dataset_matches_training_evaluation(tmp_path):
    seed = 99
    result = run(seed, tmp_path, num_hosts=_NUM_HOSTS, duration_minutes=_DURATION_MINUTES)

    gen_config = GenerationConfig(seed=seed, num_hosts=_NUM_HOSTS, duration_minutes=_DURATION_MINUTES)
    df, _ = generate_dataset(gen_config)
    _, _, test_df = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=seed)

    model, preprocessor, metadata = load_model_for_inference(tmp_path / "artifact")
    scored = score_dataset(model, preprocessor, test_df)

    assert "model_score" in scored.columns
    assert "model_pred" in scored.columns
    assert scored["model_score"].between(0, 1).all()
    assert metadata["input_dim"] > 0

    y_test = extract_labels(test_df)
    independent_metrics = evaluate_predictions(y_test, scored["model_pred"].to_numpy())

    # Inference run through score_dataset must reproduce the exact metrics
    # recorded during training's own held-out evaluation -- same model,
    # same preprocessor, same data, computed through an independent path.
    assert independent_metrics["accuracy"] == result["test_metrics"]["accuracy"]
    assert independent_metrics["f1"] == result["test_metrics"]["f1"]
    assert independent_metrics["confusion_matrix"] == result["test_metrics"]["confusion_matrix"]


def test_score_dataset_does_not_mutate_input(tmp_path):
    seed = 5
    run(seed, tmp_path, num_hosts=_NUM_HOSTS, duration_minutes=_DURATION_MINUTES)

    gen_config = GenerationConfig(seed=seed, num_hosts=_NUM_HOSTS, duration_minutes=_DURATION_MINUTES)
    df, _ = generate_dataset(gen_config)
    _, _, test_df = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=seed)

    model, preprocessor, _ = load_model_for_inference(tmp_path / "artifact")
    original_columns = list(test_df.columns)
    score_dataset(model, preprocessor, test_df)
    assert list(test_df.columns) == original_columns
