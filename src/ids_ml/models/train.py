"""Reproducible training loop for the baseline MLP classifier.

Validation-based early stopping and optional class-weighting for imbalance.
Training logic has no knowledge of Streamlit/UI -- it is driven purely by
arrays and a config, and emits portable artifacts (see save_artifact).
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from ids_ml.data.schema import SCHEMA_VERSION
from ids_ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ids_ml.models.classifier import MLPClassifier


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    seed: int
    hidden_dim: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    class_weight: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    *,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, list[dict]]:
    set_seed(config.seed)

    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=config.hidden_dim, dropout=config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    pos_weight = None
    if config.class_weight:
        n_pos = max(int(y_train.sum()), 1)
        n_neg = max(len(y_train) - n_pos, 1)
        pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

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
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).item())
            val_preds = (torch.sigmoid(val_logits) >= 0.5).int().numpy()
        val_f1 = float(f1_score(y_val, val_preds, zero_division=0))

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


def train_multiclass_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    num_classes: int,
    *,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Attack-type classifier: benign + each attack family, one softmax head.

    Mirrors train_model's shape (seeded, validation-based early stopping,
    portable via save_artifact) but with CrossEntropyLoss/argmax in place of
    BCEWithLogitsLoss/sigmoid-threshold, since this is a >2-way classification.
    """
    set_seed(config.seed)

    model = MLPClassifier(
        input_dim=X_train.shape[1], hidden_dim=config.hidden_dim, dropout=config.dropout, output_dim=num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    class_weights = None
    if config.class_weight:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0  # avoid div-by-zero for classes absent from this split
        inverse = 1.0 / counts
        class_weights = torch.tensor(inverse * (num_classes / inverse.sum()), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
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
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
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


def save_artifact(
    model: nn.Module,
    preprocessor,
    train_config: TrainConfig,
    metrics: dict,
    out_dir: Path | str,
    *,
    input_dim: int,
    output_dim: int = 1,
    class_names: list[str] | None = None,
) -> dict[str, Path]:
    """class_names is required (and output_dim must be len(class_names)) for a
    multiclass artifact; omit both for the binary normal/suspicious head."""
    from ids_ml.features import save_preprocessor  # local import avoids cycle at module load

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    preprocessor_path = out_dir / "preprocessor.pkl"
    metadata_path = out_dir / "metadata.json"

    torch.save(model.state_dict(), model_path)
    save_preprocessor(preprocessor, preprocessor_path)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "class_names": class_names,
        "numeric_features": list(getattr(preprocessor, 'encoder', preprocessor).transformers_[0][2]),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "timing_mode": getattr(preprocessor, 'timing_mode', 'absolute'),
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "train_config": dataclasses.asdict(train_config),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {"model": model_path, "preprocessor": preprocessor_path, "metadata": metadata_path}


def load_artifact(out_dir: Path | str) -> tuple[nn.Module, object, dict]:
    from ids_ml.features import load_preprocessor  # local import avoids cycle at module load

    out_dir = Path(out_dir)
    metadata = json.loads((out_dir / "metadata.json").read_text())

    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"artifact schema_version {metadata['schema_version']!r} does not match "
            f"current SCHEMA_VERSION {SCHEMA_VERSION!r}"
        )

    tc = metadata["train_config"]
    model = MLPClassifier(
        input_dim=metadata["input_dim"],
        hidden_dim=tc["hidden_dim"],
        dropout=tc["dropout"],
        output_dim=metadata.get("output_dim", 1),  # older binary-only artifacts lack this key
    )
    model.load_state_dict(torch.load(out_dir / "model.pt"))
    model.eval()

    preprocessor = load_preprocessor(out_dir / "preprocessor.pkl")
    return model, preprocessor, metadata
