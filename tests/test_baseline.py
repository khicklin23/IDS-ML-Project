from ids_ml.baseline import rule_score
from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import extract_labels
from ids_ml.inference import load_model_for_inference, score_dataset
from ids_ml.models.evaluate import evaluate_predictions
from ids_ml.pipeline import run

_SEED = 11
_NUM_HOSTS = 30
_DURATION_MINUTES = 60
_CONFIG = GenerationConfig(seed=_SEED, num_hosts=_NUM_HOSTS, duration_minutes=_DURATION_MINUTES)


def test_rule_baseline_misses_beaconing_but_catches_bursts():
    """Documents the rule baseline's known blind spot, not a flaw to fix here."""
    df, _ = generate_dataset(_CONFIG)
    scored = rule_score(df)

    beaconing = scored[scored["scenario"] == "beaconing"]
    port_scan = scored[scored["scenario"] == "port_scan"]
    auth = scored[scored["scenario"] == "auth_bruteforce"]

    assert beaconing["rule_pred"].mean() < 0.05
    assert port_scan["rule_pred"].mean() > 0.9
    assert auth["rule_pred"].mean() > 0.2  # catches some bursts, far from complete


def test_model_outperforms_rule_baseline_overall(tmp_path):
    """The model is the intended primary detector; the rule baseline is only
    a comparison signal and should measurably lose to the model, especially
    on recall since it cannot see beaconing at all."""
    result = run(_SEED, tmp_path, num_hosts=_NUM_HOSTS, duration_minutes=_DURATION_MINUTES)
    assert result["test_metrics"]["n"] > 0

    df, _ = generate_dataset(_CONFIG)
    _, _, test_df = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=_SEED)
    y_test = extract_labels(test_df)

    model, preprocessor, _ = load_model_for_inference(tmp_path / "artifact")
    model_scored = score_dataset(model, preprocessor, test_df)
    model_metrics = evaluate_predictions(y_test, model_scored["model_pred"].to_numpy())

    rule_scored = rule_score(test_df)
    rule_metrics = evaluate_predictions(y_test, rule_scored["rule_pred"].to_numpy())

    assert model_metrics["f1"] >= rule_metrics["f1"]
    assert model_metrics["recall"] > rule_metrics["recall"]
