"""Robustness evaluation: how a trained model performs beyond the exact
distribution it was trained and held-out-tested on.

Two distinct questions, both called out by docs/PRODUCT_VISION.md
("performance across unseen scenarios and changed traffic distributions"):

1. Ordinary distribution shift -- a different seed/config of the *same*
   scenario generators. Tests whether the model learned general attack
   signatures or overfit to one generation run's specifics.
2. Adversarial/evasion shift -- `stealth_mode=True` attack variants (see
   ids_ml.data.scenarios). Tests whether the model's signal is robust to an
   attacker deliberately slowing down/blending in, not just pattern-matching
   the "loud" version of an attack it happened to train on.

Both are real questions a mature IDS evaluation needs to answer honestly,
including when the answer is "it doesn't generalize well" -- that is itself
useful information, not a failure to hide.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from ids_ml.data.generator import GenerationConfig, generate_dataset
from ids_ml.features import extract_labels, extract_scenario_labels, transform_features
from ids_ml.models.evaluate import evaluate, evaluate_multiclass


def evaluate_binary_on_shifted_distribution(
    model, preprocessor, shift_config: GenerationConfig, *, threshold: float = 0.5
) -> dict:
    df, _ = generate_dataset(shift_config)
    X = transform_features(preprocessor, df)
    y = extract_labels(df)
    return {
        "shift_config": dataclasses.asdict(shift_config),
        "row_count": len(df),
        "metrics": evaluate(model, X, y, threshold=threshold),
    }


def evaluate_multiclass_on_shifted_distribution(
    model, preprocessor, class_names: list[str], shift_config: GenerationConfig
) -> dict:
    df, _ = generate_dataset(shift_config)
    X = transform_features(preprocessor, df)
    y = extract_scenario_labels(df)
    return {
        "shift_config": dataclasses.asdict(shift_config),
        "row_count": len(df),
        "metrics": evaluate_multiclass(model, X, y, class_names),
    }


def evaluate_multiclass_on_dataframe(model, preprocessor, class_names: list[str], df: pd.DataFrame) -> dict:
    """Same as evaluate_multiclass_on_shifted_distribution but for an
    already-generated DataFrame (e.g. a held-out stealth test split)."""
    X = transform_features(preprocessor, df)
    y = extract_scenario_labels(df)
    return {"row_count": len(df), "metrics": evaluate_multiclass(model, X, y, class_names)}
