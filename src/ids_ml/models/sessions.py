"""Session-level sequence preparation for the v3 (GRU) model.

Each session_id is 100% one scenario by construction (see
ids_ml.data.generator). This turns a flow DataFrame into padded
per-session sequences of the *same* preprocessed features v1/v2 use, so
the tasks/model-versions comparison is apples-to-apples on feature
representation -- only the architecture (per-flow vs per-session-sequence)
differs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ids_ml.features import transform_features


def build_session_sequences(
    df: pd.DataFrame,
    preprocessor,
    class_names: list[str],
    *,
    max_len: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (X_seq, lengths, y_seq, session_ids).

    X_seq: float32 (num_sessions, max_len, num_features), zero-padded.
    lengths: int64 (num_sessions,) -- true length, capped at max_len
        (long benign host-sessions are truncated to their first max_len
        flows chronologically; attack episodes are rarely this long).
    y_seq: int64 (num_sessions,) -- the session's scenario class index.
    session_ids: list[str], aligned with the rows above.
    """
    df_sorted = df.sort_values("event_time", kind="stable").reset_index(drop=True)
    X_all = transform_features(preprocessor, df_sorted)  # one vectorized transform, not per-session
    num_features = X_all.shape[1]

    grouped = df_sorted.groupby("session_id", sort=False)
    indices_by_session = grouped.indices  # session_id -> chronologically-ordered row positions
    scenario_by_session = grouped["scenario"].first()

    session_ids = list(indices_by_session.keys())
    class_to_int = {name: i for i, name in enumerate(class_names)}

    n_sessions = len(session_ids)
    X_seq = np.zeros((n_sessions, max_len, num_features), dtype=np.float32)
    lengths = np.zeros(n_sessions, dtype=np.int64)
    y_seq = np.zeros(n_sessions, dtype=np.int64)

    for i, session_id in enumerate(session_ids):
        rows = indices_by_session[session_id]
        n = min(len(rows), max_len)
        X_seq[i, :n, :] = X_all[rows[:n]]
        lengths[i] = n
        y_seq[i] = class_to_int[str(scenario_by_session[session_id])]

    return X_seq, lengths, y_seq, session_ids
