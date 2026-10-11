"""v3: a small GRU over a session's ordered flows, instead of classifying
one flow at a time (v1/v2 -- see ids_ml.models.classifier/train).

Directly motivated by tasks/model-maturity's open-set finding: beaconing's
signal (regular timing across many flows) is invisible to any single-flow
feature set, however augmented. This is the next rung on the same ladder
("small baseline, compare complexity" -- docs/PRODUCT_VISION.md), not a
replacement for v1/v2: all three get evaluated on the same session-level
axis in tasks/model-versions.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceClassifier(nn.Module):
    """One GRU layer over a padded flow sequence, final hidden state -> softmax head."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1])  # raw logits


@dataclasses.dataclass(frozen=True)
class SequenceTrainConfig:
    seed: int
    hidden_dim: int = 32
    lr: float = 1e-3
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 32
    class_weight: bool = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_sequence_model(
    X_train: np.ndarray,
    lengths_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    lengths_val: np.ndarray,
    y_val: np.ndarray,
    config: SequenceTrainConfig,
    num_classes: int,
    *,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Same shape as models.train.train_multiclass_model (seeded,
    validation-based early stopping, history log) but batched over
    sessions with variable-length packed sequences instead of flat rows.
    """
    _set_seed(config.seed)

    model = SequenceClassifier(input_dim=X_train.shape[-1], num_classes=num_classes, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    class_weights = None
    if config.class_weight:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        inverse = 1.0 / counts
        class_weights = torch.tensor(inverse * (num_classes / inverse.sum()), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    lengths_train_t = torch.tensor(lengths_train, dtype=torch.int64)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    lengths_val_t = torch.tensor(lengths_val, dtype=torch.int64)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    generator = torch.Generator().manual_seed(config.seed)
    n = X_train_t.shape[0]

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(config.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb, lb, yb = X_train_t[idx], lengths_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t, lengths_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).item())
            val_preds = val_logits.argmax(dim=-1).numpy()
        val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1}
        history.append(entry)
        if on_epoch is not None:
            on_epoch(entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    history.append({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
    return model, history


def predict_sequence_proba(model: nn.Module, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32), torch.tensor(lengths, dtype=torch.int64))
        return torch.softmax(logits, dim=-1).numpy()


def evaluate_sequence_multiclass(
    model: nn.Module, X: np.ndarray, lengths: np.ndarray, y: np.ndarray, class_names: list[str]
) -> dict:
    from ids_ml.models.evaluate import multiclass_metrics_from_probs

    probs = predict_sequence_proba(model, X, lengths)
    return multiclass_metrics_from_probs(probs, y, class_names)
