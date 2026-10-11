"""Feature preparation: turns schema-valid flow rows into model-ready arrays.

Kept separate from both data generation and training per the project's
module-separation requirement. Preprocessing must be fit on training data
only -- callers are responsible for passing a train-only DataFrame to
build_preprocessor.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ids_ml.data.schema import FEATURE_COLUMNS, SCENARIOS, TARGET_COLUMN

CATEGORICAL_FEATURES = ("protocol", "service", "outcome")
NUMERIC_FEATURES = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES)

LABEL_TO_INT = {"normal": 0, "suspicious": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Multiclass attack-type target: benign + every attack family in the schema's
# scenario registry, in SCENARIOS' fixed order so class indices are stable
# across runs/artifacts regardless of which scenarios happen to appear in a
# given generated dataset. `scenario` is otherwise generator-only provenance
# (see schema.NON_FEATURE_COLUMNS) -- it only becomes a *target* here, never
# a model input.
SCENARIO_CLASSES = tuple(SCENARIOS)
SCENARIO_TO_INT = {name: i for i, name in enumerate(SCENARIO_CLASSES)}
INT_TO_SCENARIO = {v: k for k, v in SCENARIO_TO_INT.items()}


def relative_timing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize gaps within complete, closed sessions; never reads labels.

    This is batch/session inference, not an online transform for partial flows.
    Keep absolute event times in the original telemetry for display and replay.
    """
    if 'session_id' not in df:
        raise ValueError('Relative timing requires complete sessions and session_id.')
    if df.session_id.isna().any():
        raise ValueError('Relative timing requires a session_id for every flow.')
    out = df.copy()
    gaps = out.prior_gap_s.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError('Timing gaps must be finite and nonnegative.')
    for indices in out.groupby('session_id', sort=False).indices.values():
        positive = np.sort(gaps[indices][gaps[indices] > 0])
        scale = float(np.mean(positive)) if len(positive) else 1.
        gaps[indices] /= max(scale, 1e-6)
    out['prior_gap_s'] = gaps
    return out


class SessionTimingPreprocessor:
    """Persist the closed-session transform with its train-fitted encoder.

    hybrid_log compresses only the absolute gap with log1p; the relative gap
    is calculated from the original seconds, before that compression.
    """

    timing_mode = 'relative'  # Backward-compatible with early relative-only pickles.

    def __init__(self, encoder, timing_mode='relative'):
        self.encoder = encoder
        self.timing_mode = timing_mode

    def transform(self, df):
        prepared = relative_timing_frame(df)
        columns = list(FEATURE_COLUMNS)
        if self.timing_mode in ('hybrid', 'hybrid_log'):
            prepared = df.assign(session_relative_gap=prepared.prior_gap_s)
            columns.append('session_relative_gap')
            if self.timing_mode == 'hybrid_log':
                prepared = prepared.assign(prior_gap_s=np.log1p(prepared.prior_gap_s))
        return self.encoder.transform(prepared[columns])

    def get_feature_names_out(self):
        return self.encoder.get_feature_names_out()


def build_preprocessor(train_df: pd.DataFrame, *, timing_mode: str = 'absolute') -> ColumnTransformer | SessionTimingPreprocessor:
    """Fit on training data only. Relative mode requires complete closed sessions."""
    if timing_mode not in ('absolute', 'relative', 'hybrid', 'hybrid_log'):
        raise ValueError('timing_mode must be absolute, relative, hybrid, or hybrid_log.')
    fit_df = relative_timing_frame(train_df) if timing_mode != 'absolute' else train_df
    numeric = list(NUMERIC_FEATURES)
    columns = list(FEATURE_COLUMNS)
    if timing_mode in ('hybrid', 'hybrid_log'):
        fit_df = train_df.assign(session_relative_gap=fit_df.prior_gap_s)
        if timing_mode == 'hybrid_log':
            fit_df = fit_df.assign(prior_gap_s=np.log1p(fit_df.prior_gap_s))
        numeric.append('session_relative_gap')
        columns.append('session_relative_gap')
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    preprocessor.fit(fit_df[columns])
    return SessionTimingPreprocessor(preprocessor, timing_mode) if timing_mode != 'absolute' else preprocessor


def transform_features(preprocessor: ColumnTransformer | SessionTimingPreprocessor, df: pd.DataFrame) -> np.ndarray:
    frame = df if isinstance(preprocessor, SessionTimingPreprocessor) else df[list(FEATURE_COLUMNS)]
    return preprocessor.transform(frame).astype(np.float32)


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COLUMN].astype(str).map(LABEL_TO_INT).to_numpy(dtype=np.int64)


def extract_labels_for_classes(df: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Multiclass target restricted to an arbitrary class list (e.g. a subset
    of SCENARIO_CLASSES, for training an open-set model that deliberately
    excludes one attack family -- see ids_ml.openset). Raises on any
    scenario value not in class_names, same as extract_scenario_labels.
    """
    mapping = {name: i for i, name in enumerate(class_names)}
    mapped = df["scenario"].astype(str).map(mapping)
    if mapped.isna().any():
        unknown = sorted(set(df.loc[mapped.isna(), "scenario"].astype(str)))
        raise ValueError(f"scenario value(s) not in class_names: {unknown}")
    return mapped.to_numpy(dtype=np.int64)


def extract_scenario_labels(df: pd.DataFrame) -> np.ndarray:
    """Multiclass attack-type target (benign + each attack family)."""
    return extract_labels_for_classes(df, list(SCENARIO_CLASSES))


def save_preprocessor(preprocessor: ColumnTransformer | SessionTimingPreprocessor, path: Path | str) -> None:
    Path(path).write_bytes(pickle.dumps(preprocessor))


def load_preprocessor(path: Path | str) -> ColumnTransformer | SessionTimingPreprocessor:
    return pickle.loads(Path(path).read_bytes())
