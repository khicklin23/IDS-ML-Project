"""v3 sequence model: session-sequence prep + GRU training/eval, in isolation
before the full v1/v2/v3 comparison harness (tasks/model-versions)."""

import numpy as np

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import SCENARIO_CLASSES, build_preprocessor
from ids_ml.models.sequence import SequenceTrainConfig, evaluate_sequence_multiclass, train_sequence_model
from ids_ml.models.sessions import build_session_sequences

_CLASS_NAMES = list(SCENARIO_CLASSES)
_CONFIG = GenerationConfig(
    seed=99, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
    port_scan_episodes=6, auth_bruteforce_episodes=6, beaconing_episodes=6,
    service_flood_episodes=6, outbound_transfer_episodes=6,
)


def _prepared_sequences(max_len=50):
    df, _ = generate_dataset(_CONFIG)
    train_df, val_df, test_df = split_dataset(df, val_frac=0.2, test_frac=0.2, seed=99)
    preprocessor = build_preprocessor(train_df)

    train_seq = build_session_sequences(train_df, preprocessor, _CLASS_NAMES, max_len=max_len)
    val_seq = build_session_sequences(val_df, preprocessor, _CLASS_NAMES, max_len=max_len)
    test_seq = build_session_sequences(test_df, preprocessor, _CLASS_NAMES, max_len=max_len)
    return train_seq, val_seq, test_seq


def test_build_session_sequences_shapes_and_coverage():
    (X, lengths, y, session_ids), _, _ = _prepared_sequences()

    assert X.shape[0] == len(lengths) == len(y) == len(session_ids)
    assert X.shape[1] == 50
    assert (lengths >= 1).all()
    assert (lengths <= 50).all()
    # every class present among the sessions (sanity -- session-level, not row-level)
    assert set(y) == set(range(len(_CLASS_NAMES)))
    # a session's sequence should be all zeros past its true length
    for i in range(min(10, len(lengths))):
        n = lengths[i]
        if n < X.shape[1]:
            assert np.allclose(X[i, n:, :], 0.0)


def test_sequence_model_trains_and_evaluates_reasonably():
    (X_train, len_train, y_train, _), (X_val, len_val, y_val, _), (X_test, len_test, y_test, _) = _prepared_sequences()

    config = SequenceTrainConfig(seed=0, max_epochs=30, patience=8)
    model, history = train_sequence_model(
        X_train, len_train, y_train, X_val, len_val, y_val, config, len(_CLASS_NAMES)
    )
    assert len(history) > 1

    metrics = evaluate_sequence_multiclass(model, X_test, len_test, y_test, _CLASS_NAMES)
    assert metrics["macro_f1"] > 0.5, metrics
    assert metrics["n"] == len(y_test)
