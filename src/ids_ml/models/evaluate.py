"""Held-out evaluation for the baseline classifier.

Synthetic evaluation results describe performance on generated data only
and must not be presented as evidence of real-world detection accuracy
(see docs/PRODUCT_VISION.md).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def predict_class_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Multiclass softmax probabilities, shape (n, num_classes)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).numpy()


def evaluate_predictions(y: np.ndarray, preds: np.ndarray, *, threshold: float | None = None) -> dict:
    """Shared metric yardstick so any predictor (model or rule baseline) is scored identically."""
    preds = np.asarray(preds).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=(0, 1)).ravel()
    accuracy = float((preds == y).mean())
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
    }


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, *, threshold: float = 0.5) -> dict:
    probs = predict_proba(model, X)
    preds = (probs >= threshold).astype(np.int64)
    return evaluate_predictions(y, preds, threshold=threshold)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> dict:
    """Standard ECE: bin predictions by confidence (max softmax prob), compare
    each bin's average confidence to its actual accuracy. A well-calibrated
    model's "I'm 90% sure" predictions should be right ~90% of the time;
    ECE is the weighted average gap between confidence and accuracy.

    Reported for transparency, not asserted "good" -- see
    tasks/model-maturity: this model is measurably overconfident (same root
    cause that made naive open-set rejection fail in ids_ml.openset), and
    that's an honest finding, not something to hide behind a passing test.
    """
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    bins = []
    n = len(confidences)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
        bins.append({
            "bin_range": [float(bin_edges[b]), float(bin_edges[b + 1])],
            "confidence": bin_confidence,
            "accuracy": bin_accuracy,
            "count": count,
        })

    return {"ece": float(ece), "n_bins": n_bins, "bins": bins, "n": n}


def multiclass_metrics_from_predictions(preds: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    """Shared scoring logic for any multiclass predictor, given discrete
    integer predictions directly -- used when there's no softmax
    probability to argmax (e.g. majority-vote session rollup in
    tasks/model-versions). multiclass_metrics_from_probs below is a thin
    wrapper for the probability case.
    """
    num_classes = len(class_names)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, preds, labels=labels, average=None, zero_division=0
    )
    macro_f1 = float(f1.mean())
    accuracy = float((preds == y).mean())
    cm = confusion_matrix(y, preds, labels=labels)

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }

    return {
        "class_names": list(class_names),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y)),
    }


def multiclass_metrics_from_probs(probs: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    return multiclass_metrics_from_predictions(probs.argmax(axis=-1), y, class_names)


def evaluate_multiclass(
    model: nn.Module, X: np.ndarray, y: np.ndarray, class_names: list[str]
) -> dict:
    """Per-class precision/recall/F1 + full confusion matrix for the attack-type head."""
    probs = predict_class_proba(model, X)
    return multiclass_metrics_from_probs(probs, y, class_names)
