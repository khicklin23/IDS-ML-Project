"""Open-set recognition: flag "this doesn't match anything I was trained on"
instead of confidently (and wrongly) picking the closest known class.

This is deliberately the highest-priority item in tasks/model-maturity: a
rule-based system can only ever match patterns someone wrote down, so a
genuinely novel attack is invisible to it by construction. A classifier
with confidence-based rejection can flag novelty it was never shown -- a
real capability a NN has that hand-written rules structurally cannot,
which is exactly the kind of thing this project exists to demonstrate (see
docs/PRODUCT_VISION.md and the user-priority note in context/roadmap).

Method: standard max-softmax-probability thresholding (Hendrycks & Gimpel
2017) -- the simplest established open-set-recognition baseline, not a
novel technique. Good enough to demonstrate the capability; more advanced
methods (e.g. a learned rejection head, distance-in-feature-space) are
future work if this baseline isn't sufficient in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ids_ml.features import transform_features
from ids_ml.models.evaluate import predict_class_proba

UNKNOWN_LABEL = "unknown"


def classify_with_rejection(probs: np.ndarray, class_names: list[str], threshold: float) -> list[str]:
    """argmax class name, or UNKNOWN_LABEL if its probability is below threshold."""
    idx = probs.argmax(axis=-1)
    maxp = probs.max(axis=-1)
    return [class_names[i] if p >= threshold else UNKNOWN_LABEL for i, p in zip(idx, maxp)]


def evaluate_open_set(
    model,
    preprocessor,
    test_df: pd.DataFrame,
    known_class_names: list[str],
    threshold: float,
) -> dict:
    """Split test_df into known-class and novel-class (scenario not in
    known_class_names) rows and report, separately:

    - novel_detection_rate: fraction of truly novel rows correctly flagged
      UNKNOWN_LABEL (higher is better -- this is the actual capability).
    - known_false_rejection_rate: fraction of known-class rows incorrectly
      flagged UNKNOWN_LABEL (the cost of the threshold -- lower is better).
    - known_accuracy: accuracy on known-class rows (an UNKNOWN_LABEL
      prediction counts as wrong here, since it is a missed detection).
    """
    X = transform_features(preprocessor, test_df)
    probs = predict_class_proba(model, X)
    preds = classify_with_rejection(probs, known_class_names, threshold)

    true = test_df["scenario"].astype(str).to_numpy()
    is_novel = ~np.isin(true, known_class_names)
    is_known = ~is_novel

    novel_count = int(is_novel.sum())
    known_count = int(is_known.sum())
    novel_flagged_unknown = sum(1 for pred, novel in zip(preds, is_novel) if novel and pred == UNKNOWN_LABEL)
    known_flagged_unknown = sum(1 for pred, known in zip(preds, is_known) if known and pred == UNKNOWN_LABEL)
    known_correct = sum(1 for pred, t, known in zip(preds, true, is_known) if known and pred == t)

    return {
        "threshold": threshold,
        "known_class_names": list(known_class_names),
        "novel_scenarios": sorted(set(true[is_novel])) if novel_count else [],
        "novel_count": novel_count,
        "novel_detection_rate": (novel_flagged_unknown / novel_count) if novel_count else None,
        "known_count": known_count,
        "known_accuracy": (known_correct / known_count) if known_count else None,
        "known_false_rejection_rate": (known_flagged_unknown / known_count) if known_count else None,
    }
