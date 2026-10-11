"""Streamlit session-state orchestration for the IDS workbench.

Deliberately not using st.cache_data/cache_resource: generation/training are
triggered by explicit buttons (matching the vision's "Generate"/"Train"
workflow), and results persist in st.session_state across page navigation
within one browser session until the user explicitly regenerates.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from ids_ml.alerts import generate_alerts_from_scores, group_into_incidents, init_db, insert_alerts
from ids_ml.baseline import rule_score
from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import build_preprocessor, extract_labels, transform_features
from ids_ml.inference import score_dataset
from ids_ml.models.evaluate import evaluate, evaluate_predictions
from ids_ml.models.train import TrainConfig, train_model
from ids_ml.replay import ReplayController

DEFAULT_GEN_CONFIG = GenerationConfig(
    seed=2024,
    num_hosts=60,
    duration_minutes=180.0,
    background_rate_per_min=60.0,
    # 6 episodes/family (not 4): split_dataset groups by session_id without
    # label stratification, and at 4 episodes this exact seed's test split
    # landed zero suspicious rows -- a first-time user's default click-through
    # would hit a 0-recall Evaluation page. Verified empirically (see
    # tasks/streamlit-ui) that 6 gives a healthy suspicious count in both
    # val and test for this seed.
    port_scan_episodes=6,
    auth_bruteforce_episodes=6,
    beaconing_episodes=6,
)
DEFAULT_TRAIN_CONFIG = TrainConfig(seed=2024)
DEFAULT_THRESHOLD = 0.5

_STATE_DEFAULTS: dict[str, object] = {
    "gen_config": DEFAULT_GEN_CONFIG,
    "dataset": None,
    "splits": None,
    "train_config": DEFAULT_TRAIN_CONFIG,
    "artifact": None,
    "threshold": DEFAULT_THRESHOLD,
    "scored_test": None,
    "alert_conn": None,
    "replay": None,
    "split_warning": None,
}


def init_session_state() -> None:
    for key, value in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_dataset() -> bool:
    return st.session_state.get("dataset") is not None


def has_artifact() -> bool:
    return st.session_state.get("artifact") is not None


def has_scores() -> bool:
    return st.session_state.get("scored_test") is not None


def generate(config: GenerationConfig) -> None:
    df, manifest = generate_dataset(config)
    splits = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=config.seed)
    st.session_state.gen_config = config
    st.session_state.dataset = (df, manifest)
    st.session_state.splits = splits
    st.session_state.artifact = None
    st.session_state.scored_test = None
    st.session_state.replay = None
    st.session_state.alert_conn = None

    # split_dataset groups by session_id without label stratification, so a
    # small/unlucky config can land zero suspicious rows in val or test by
    # chance -- caught empirically while building this page (see
    # tasks/streamlit-ui). Surface it rather than let training/evaluation
    # silently show degenerate 0-recall metrics with no explanation.
    st.session_state.split_warning = None
    for name, split_df in (("validation", splits[1]), ("test", splits[2])):
        if (split_df["label"] == "suspicious").sum() == 0:
            st.session_state.split_warning = (
                f"The {name} split landed zero suspicious rows by chance (the session-grouped "
                "split isn't label-stratified). Metrics on that split will look degenerate. "
                "Try a different seed or more attack episodes."
            )
            break


def train(train_config: TrainConfig, *, on_epoch: Callable[[dict], None] | None = None) -> None:
    train_df, val_df, test_df = st.session_state.splits

    preprocessor = build_preprocessor(train_df)
    X_train, y_train = transform_features(preprocessor, train_df), extract_labels(train_df)
    X_val, y_val = transform_features(preprocessor, val_df), extract_labels(val_df)
    X_test, y_test = transform_features(preprocessor, test_df), extract_labels(test_df)

    model, history = train_model(X_train, y_train, X_val, y_val, train_config, on_epoch=on_epoch)
    test_metrics = evaluate(model, X_test, y_test, threshold=st.session_state.threshold)

    st.session_state.train_config = train_config
    st.session_state.artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "history": history,
        "test_metrics": test_metrics,
        "input_dim": X_train.shape[1],
    }
    st.session_state.scored_test = None
    st.session_state.replay = None


def score(*, threshold: float | None = None) -> None:
    artifact = st.session_state.artifact
    _, _, test_df = st.session_state.splits
    t = st.session_state.threshold if threshold is None else threshold
    st.session_state.threshold = t

    scored = score_dataset(artifact["model"], artifact["preprocessor"], test_df, threshold=t)
    scored = rule_score(scored)
    st.session_state.scored_test = scored
    st.session_state.replay = None

    y_test = extract_labels(test_df)
    artifact["test_metrics"] = evaluate_predictions(y_test, scored["model_pred"].to_numpy(), threshold=t)
    artifact["rule_metrics"] = evaluate_predictions(y_test, scored["rule_pred"].to_numpy())


def get_alert_conn():
    if st.session_state.alert_conn is None:
        st.session_state.alert_conn = init_db(":memory:", check_same_thread=False)
    return st.session_state.alert_conn


def refresh_alerts() -> None:
    """Rebuild the in-memory alert store from the current scored test set."""
    conn = get_alert_conn()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    run_id = st.session_state.gen_config.run_id
    alerts = generate_alerts_from_scores(
        st.session_state.scored_test, run_id, threshold=st.session_state.threshold
    )
    insert_alerts(conn, alerts)
    group_into_incidents(conn, run_id)


def get_replay() -> ReplayController:
    if st.session_state.replay is None:
        st.session_state.replay = ReplayController(st.session_state.scored_test, speed=1.0)
    return st.session_state.replay
