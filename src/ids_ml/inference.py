"""Load a trained artifact and score flow data with it.

Separate from training (ids_ml.models.train) and from the rule baseline
(ids_ml.baseline): this module's only job is "artifact in, scores out," so
replay and alert-generation code depend on a narrow, stable interface
instead of training internals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn

from ids_ml.features import transform_features
from ids_ml.models.evaluate import predict_class_proba, predict_proba
from ids_ml.models.train import load_artifact


def load_model_for_inference(artifact_dir: Path | str) -> tuple[nn.Module, object, dict]:
    """Thin, named wrapper over train.load_artifact for call sites outside training."""
    return load_artifact(artifact_dir)


def score_dataset(
    model: nn.Module,
    preprocessor,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a copy of df with model_score (P(suspicious)) and model_pred columns.

    The model is the primary detector here: model_pred is what the rest of
    the pipeline (alerts, replay) should treat as "the detection," with any
    rule-based baseline kept as a separate comparison column, never a
    substitute for this one.
    """
    X = transform_features(preprocessor, df)
    scores = predict_proba(model, X)
    out = df.copy()
    out["model_score"] = scores.astype(np.float32)
    out["model_pred"] = (scores >= threshold).astype(np.int64)
    return out


def score_dataset_multiclass(
    model: nn.Module,
    preprocessor,
    df: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Return a copy of df with the model's own predicted attack type.

    predicted_scenario is the model's inference, not ground truth -- keep it
    visually distinct from the generator's `scenario` column wherever both
    are shown (see theme.ground_truth_badge/inference_badge in the UI).
    """
    X = transform_features(preprocessor, df)
    probs = predict_class_proba(model, X)
    pred_idx = probs.argmax(axis=-1)
    out = df.copy()
    out["predicted_scenario"] = [class_names[i] for i in pred_idx]
    out["predicted_scenario_confidence"] = probs.max(axis=-1).astype(np.float32)
    return out
