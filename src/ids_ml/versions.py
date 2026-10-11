"""Model version registry: v1 -> v2 -> v3, compared on one shared harness
so progression is measurable data, not prose (tasks/model-versions).

- v1: per-flow MLP (ids_ml.models.classifier), trained on one normal
  (non-stealth) generation run. The original attack-type-classifier
  baseline.
- v2: identical architecture to v1. Different training recipe only:
  normal + stealth examples mixed into training (validated in
  tests/test_robustness.py to recover most stealth-evasion robustness
  without hurting normal-case performance). Isolates "better training
  data" as the improvement axis.
- v3: session-level GRU (ids_ml.models.sequence) instead of a per-flow
  classifier, trained on the SAME normal-only recipe as v1 (same
  preprocessor, same data) so the comparison isolates "better
  architecture" as a *different* improvement axis from v2's.

All three are scored on the same axis -- session-level accuracy/macro-F1 --
by rolling v1/v2's per-flow predictions up to one prediction per session
(majority vote). v1/v2 additionally report their native per-flow metrics,
since that's the finer-grained thing they were actually built to do.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import torch

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.features import SCENARIO_CLASSES, build_preprocessor, extract_scenario_labels, transform_features
from ids_ml.inference import score_dataset_multiclass
from ids_ml.models.evaluate import evaluate_multiclass, multiclass_metrics_from_predictions
from ids_ml.models.sequence import SequenceTrainConfig, evaluate_sequence_multiclass, train_sequence_model
from ids_ml.models.sessions import build_session_sequences
from ids_ml.models.train import TrainConfig, save_artifact, train_multiclass_model

VERSIONS_SCHEMA = "1.0.0"
CLASS_NAMES = list(SCENARIO_CLASSES)


def session_level_from_flow_model(model, preprocessor, df: pd.DataFrame, class_names: list[str]) -> dict:
    """Roll a per-flow model's predictions up to one prediction per session
    (majority vote across that session's flows), for a fair comparison
    against v3's native session-level output.
    """
    scored = score_dataset_multiclass(model, preprocessor, df, class_names)
    pred_mode = scored.groupby("session_id")["predicted_scenario"].agg(lambda s: s.value_counts().idxmax())
    true_first = scored.groupby("session_id")["scenario"].first().astype(str)
    aligned_pred = pred_mode.reindex(true_first.index)

    name_to_int = {name: i for i, name in enumerate(class_names)}
    y_true = true_first.map(name_to_int).to_numpy()
    y_pred = aligned_pred.map(name_to_int).to_numpy()
    return multiclass_metrics_from_predictions(y_pred, y_true, class_names)


def run_comparison(
    out_dir: Path | str,
    *,
    seed: int = 7,
    num_hosts: int = 150,
    duration_minutes: float = 180.0,
    episodes_per_family: int = 50,
    train_config: TrainConfig | None = None,
    sequence_config: SequenceTrainConfig | None = None,
    max_seq_len: int = 50,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_config = train_config or TrainConfig(seed=seed)
    sequence_config = sequence_config or SequenceTrainConfig(seed=seed)

    normal_config = GenerationConfig(
        seed=seed, num_hosts=num_hosts, duration_minutes=duration_minutes, background_rate_per_min=60.0,
        port_scan_episodes=episodes_per_family, auth_bruteforce_episodes=episodes_per_family,
        beaconing_episodes=episodes_per_family, service_flood_episodes=episodes_per_family,
        outbound_transfer_episodes=episodes_per_family, stealth_mode=False,
    )
    stealth_config = dataclasses.replace(normal_config, seed=seed + 1000, stealth_mode=True)

    normal_df, normal_manifest = generate_dataset(normal_config)
    normal_train, normal_val, normal_test = split_dataset(normal_df, val_frac=0.15, test_frac=0.15, seed=seed)

    stealth_df, stealth_manifest = generate_dataset(stealth_config)
    stealth_train, stealth_val, stealth_test = split_dataset(
        stealth_df, val_frac=0.15, test_frac=0.15, seed=seed + 1000
    )

    versions: dict[str, dict] = {}

    # ---------------- v1: per-flow MLP, normal-only training ----------------
    v1_preprocessor = build_preprocessor(normal_train)
    X_train = transform_features(v1_preprocessor, normal_train)
    y_train = extract_scenario_labels(normal_train)
    X_val = transform_features(v1_preprocessor, normal_val)
    y_val = extract_scenario_labels(normal_val)
    v1_model, v1_history = train_multiclass_model(X_train, y_train, X_val, y_val, train_config, len(CLASS_NAMES))

    X_test = transform_features(v1_preprocessor, normal_test)
    y_test = extract_scenario_labels(normal_test)
    v1_normal_flow = evaluate_multiclass(v1_model, X_test, y_test, CLASS_NAMES)
    v1_normal_session = session_level_from_flow_model(v1_model, v1_preprocessor, normal_test, CLASS_NAMES)
    v1_stealth_session = session_level_from_flow_model(v1_model, v1_preprocessor, stealth_test, CLASS_NAMES)

    save_artifact(
        v1_model, v1_preprocessor, train_config, v1_normal_flow, out_dir / "v1",
        input_dim=X_train.shape[1], output_dim=len(CLASS_NAMES), class_names=CLASS_NAMES,
    )
    versions["v1"] = {
        "description": "per-flow MLP, normal-only training",
        "architecture": "MLP (per-flow)",
        "training_recipe": "normal-only",
        "normal_flow_metrics": v1_normal_flow,
        "normal_session_metrics": v1_normal_session,
        "stealth_session_metrics": v1_stealth_session,
        "history": v1_history,
    }

    # ---------------- v2: per-flow MLP, normal+stealth mixed training ----------------
    mixed_train = pd.concat([normal_train, stealth_train], ignore_index=True)
    mixed_val = pd.concat([normal_val, stealth_val], ignore_index=True)
    v2_preprocessor = build_preprocessor(mixed_train)
    X_train2 = transform_features(v2_preprocessor, mixed_train)
    y_train2 = extract_scenario_labels(mixed_train)
    X_val2 = transform_features(v2_preprocessor, mixed_val)
    y_val2 = extract_scenario_labels(mixed_val)
    v2_model, v2_history = train_multiclass_model(X_train2, y_train2, X_val2, y_val2, train_config, len(CLASS_NAMES))

    X_test2 = transform_features(v2_preprocessor, normal_test)
    v2_normal_flow = evaluate_multiclass(v2_model, X_test2, y_test, CLASS_NAMES)
    v2_normal_session = session_level_from_flow_model(v2_model, v2_preprocessor, normal_test, CLASS_NAMES)
    v2_stealth_session = session_level_from_flow_model(v2_model, v2_preprocessor, stealth_test, CLASS_NAMES)

    save_artifact(
        v2_model, v2_preprocessor, train_config, v2_normal_flow, out_dir / "v2",
        input_dim=X_train2.shape[1], output_dim=len(CLASS_NAMES), class_names=CLASS_NAMES,
    )
    versions["v2"] = {
        "description": "per-flow MLP, normal+stealth mixed training",
        "architecture": "MLP (per-flow)",
        "training_recipe": "normal+stealth mixed",
        "normal_flow_metrics": v2_normal_flow,
        "normal_session_metrics": v2_normal_session,
        "stealth_session_metrics": v2_stealth_session,
        "history": v2_history,
    }

    # ---------------- v3: session-level GRU, normal-only training (same recipe as v1) ----------------
    train_seq = build_session_sequences(normal_train, v1_preprocessor, CLASS_NAMES, max_len=max_seq_len)
    val_seq = build_session_sequences(normal_val, v1_preprocessor, CLASS_NAMES, max_len=max_seq_len)
    test_seq = build_session_sequences(normal_test, v1_preprocessor, CLASS_NAMES, max_len=max_seq_len)
    stealth_test_seq = build_session_sequences(stealth_test, v1_preprocessor, CLASS_NAMES, max_len=max_seq_len)

    v3_model, v3_history = train_sequence_model(
        train_seq[0], train_seq[1], train_seq[2],
        val_seq[0], val_seq[1], val_seq[2],
        sequence_config, len(CLASS_NAMES),
    )
    v3_normal_session = evaluate_sequence_multiclass(v3_model, test_seq[0], test_seq[1], test_seq[2], CLASS_NAMES)
    v3_stealth_session = evaluate_sequence_multiclass(
        v3_model, stealth_test_seq[0], stealth_test_seq[1], stealth_test_seq[2], CLASS_NAMES
    )

    torch.save(v3_model.state_dict(), out_dir / "v3_model.pt")
    (out_dir / "v3_metadata.json").write_text(json.dumps({
        "architecture": "GRU (session-level)",
        "hidden_dim": sequence_config.hidden_dim,
        "max_seq_len": max_seq_len,
        "class_names": CLASS_NAMES,
    }, indent=2))

    versions["v3"] = {
        "description": "session-level GRU, normal-only training (same recipe as v1)",
        "architecture": "GRU (session-level)",
        "training_recipe": "normal-only",
        "normal_flow_metrics": None,
        "normal_session_metrics": v3_normal_session,
        "stealth_session_metrics": v3_stealth_session,
        "history": v3_history,
    }

    comparison = {
        "versions_schema": VERSIONS_SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "class_names": CLASS_NAMES,
        "dataset": {
            "normal_manifest": normal_manifest,
            "stealth_manifest": stealth_manifest,
            "session_counts": {
                "normal_train": len(train_seq[3]),
                "normal_val": len(val_seq[3]),
                "normal_test": len(test_seq[3]),
                "stealth_test": len(stealth_test_seq[3]),
            },
        },
        "versions": versions,
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True, default=str))
    return comparison
