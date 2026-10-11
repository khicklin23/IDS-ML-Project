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


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """Fit a ColumnTransformer on training data only."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    preprocessor.fit(train_df[list(FEATURE_COLUMNS)])
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(df[list(FEATURE_COLUMNS)]).astype(np.float32)


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


def save_preprocessor(preprocessor: ColumnTransformer, path: Path | str) -> None:
    Path(path).write_bytes(pickle.dumps(preprocessor))


def load_preprocessor(path: Path | str) -> ColumnTransformer:
    return pickle.loads(Path(path).read_bytes())
