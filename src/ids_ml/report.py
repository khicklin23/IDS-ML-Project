"""Unified precomputed report: the one artifact a presentation layer needs.

This is the backend/UI contract. Run everything once (generate -> split ->
train binary + multiclass -> score -> alerts), write every intermediate
artifact to disk, and assemble ONE report.json with stable field names that
a UI can be built against even while model quality/tuning keeps changing
underneath it. See REPORT_VERSION: bump it if this function's output shape
changes in a way that could break a UI reading it; dataset/model internals
can keep evolving without touching this number as long as these fields and
their meaning stay the same.

Not a Streamlit page and not imported by one -- this module has no UI
dependency, matching the project's module-separation rule.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ids_ml.alerts import export_alerts, generate_alerts_from_scores, group_into_incidents, init_db, insert_alerts, list_alerts
from ids_ml.baseline import rule_score
from ids_ml.data.generator import GenerationConfig, combine_runs, generate_dataset, save_dataset, split_dataset
from ids_ml.data.schema import SCHEMA_VERSION
from ids_ml.features import (
    SCENARIO_CLASSES,
    build_preprocessor,
    extract_labels,
    extract_scenario_labels,
    transform_features,
)
from ids_ml.inference import score_dataset, score_dataset_multiclass
from ids_ml.models.evaluate import (
    evaluate,
    evaluate_multiclass,
    evaluate_predictions,
    expected_calibration_error,
    multiclass_metrics_from_predictions,
    predict_class_proba,
)
from ids_ml.models.sequence import SequenceTrainConfig, predict_sequence_proba, train_sequence_model
from ids_ml.models.sessions import build_session_sequences
from ids_ml.models.train import TrainConfig, save_artifact, train_model, train_multiclass_model
from ids_ml.versions import max_session_length

REPORT_VERSION = "1.1.0"  # 1.1.0 adds multiclass_model.calibration (additive)

#: Columns shown in `dataset.sample_flows` -- the "what a normal IDS would be
#: seeing" display fields: real-looking addresses, not internal host ids.
_DISPLAY_COLUMNS = (
    "flow_id", "event_time", "src_ip", "dst_ip", "protocol", "service",
    "duration_s", "packets_sent", "packets_recv", "bytes_sent", "bytes_recv",
    "outcome", "prior_gap_s", "scenario", "label",
)


def build_report(
    seed: int,
    out_dir: Path | str,
    *,
    gen_config: GenerationConfig | None = None,
    num_hosts: int = 60,
    duration_minutes: float = 180.0,
    sample_size: int = 50,
    alert_threshold: float = 0.5,
    binary_train_config: TrainConfig | None = None,
    multiclass_train_config: TrainConfig | None = None,
) -> dict:
    out_dir = Path(out_dir)
    gen_config = gen_config or GenerationConfig(seed=seed, num_hosts=num_hosts, duration_minutes=duration_minutes)
    binary_train_config = binary_train_config or TrainConfig(seed=seed)
    multiclass_train_config = multiclass_train_config or TrainConfig(seed=seed)

    df, manifest = generate_dataset(gen_config)
    save_dataset(df, manifest, out_dir / "dataset")

    train_df, val_df, test_df = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=seed)
    preprocessor = build_preprocessor(train_df)

    X_train = transform_features(preprocessor, train_df)
    X_val = transform_features(preprocessor, val_df)
    X_test = transform_features(preprocessor, test_df)

    # --- binary normal/suspicious head + rule baseline comparison ---
    y_train_bin = extract_labels(train_df)
    y_val_bin = extract_labels(val_df)
    y_test_bin = extract_labels(test_df)

    binary_model, binary_history = train_model(X_train, y_train_bin, X_val, y_val_bin, binary_train_config)
    binary_metrics = evaluate(binary_model, X_test, y_test_bin)

    scored = score_dataset(binary_model, preprocessor, test_df, threshold=alert_threshold)
    scored = rule_score(scored)
    rule_metrics = evaluate_predictions(y_test_bin, scored["rule_pred"].to_numpy())

    save_artifact(
        binary_model, preprocessor, binary_train_config, binary_metrics,
        out_dir / "artifact_binary", input_dim=X_train.shape[1],
    )

    # --- multiclass attack-type head ---
    class_names = list(SCENARIO_CLASSES)
    y_train_mc = extract_scenario_labels(train_df)
    y_val_mc = extract_scenario_labels(val_df)
    y_test_mc = extract_scenario_labels(test_df)

    multiclass_model, multiclass_history = train_multiclass_model(
        X_train, y_train_mc, X_val, y_val_mc, multiclass_train_config, len(class_names)
    )
    multiclass_metrics = evaluate_multiclass(multiclass_model, X_test, y_test_mc, class_names)
    # Reported for transparency, not tuned to look good -- see
    # tasks/model-maturity: this model is measurably overconfident.
    calibration = expected_calibration_error(predict_class_proba(multiclass_model, X_test), y_test_mc)

    scored = score_dataset_multiclass(multiclass_model, preprocessor, scored, class_names)

    save_artifact(
        multiclass_model, preprocessor, multiclass_train_config, multiclass_metrics,
        out_dir / "artifact_multiclass", input_dim=X_train.shape[1],
        output_dim=len(class_names), class_names=class_names,
    )

    # --- alerts: binary model_pred decides "is this an alert", multiclass
    # predicted_scenario rides along as the model's attack-type guess ---
    conn = init_db(out_dir / "alerts.db")
    alerts = generate_alerts_from_scores(scored, gen_config.run_id, threshold=alert_threshold)
    insert_alerts(conn, alerts)
    group_into_incidents(conn, gen_config.run_id)
    all_alerts = list_alerts(conn, order_by="start_time")
    export_alerts(conn, out_dir / "alerts.csv", fmt="csv")

    severity_counts = all_alerts["severity"].value_counts().to_dict() if not all_alerts.empty else {}

    report = {
        "report_version": REPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": gen_config.run_id,
        "dataset": {
            "manifest": manifest,
            "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "sample_flows": df[list(_DISPLAY_COLUMNS)].head(sample_size).to_dict(orient="records"),
        },
        "binary_model": {
            "train_config": dataclasses.asdict(binary_train_config),
            "history": binary_history,
            "test_metrics": binary_metrics,
            "rule_baseline_metrics": rule_metrics,
        },
        "multiclass_model": {
            "class_names": class_names,
            "train_config": dataclasses.asdict(multiclass_train_config),
            "history": multiclass_history,
            "test_metrics": multiclass_metrics,
            "calibration": calibration,
        },
        "alerts": {
            "total": int(len(all_alerts)),
            "by_severity": {str(k): int(v) for k, v in severity_counts.items()},
            "sample": all_alerts.head(sample_size).to_dict(orient="records"),
        },
        "artifact_paths": {
            "dataset": str(out_dir / "dataset"),
            "binary_model": str(out_dir / "artifact_binary"),
            "multiclass_model": str(out_dir / "artifact_multiclass"),
            "alerts_db": str(out_dir / "alerts.db"),
            "alerts_csv": str(out_dir / "alerts.csv"),
        },
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def build_session_report(
    seed: int,
    out_dir: Path | str,
    *,
    num_hosts: int = 200,
    duration_minutes: float = 180.0,
    episodes_per_family: int = 100,
    sample_size: int = 50,
    alert_threshold: float = 0.5,
    sequence_config: SequenceTrainConfig | None = None,
    max_seq_len: int | None = None,
    timing_mode: str = 'absolute',
) -> dict:
    """Latest-version detection report: the session GRU (tasks/model-versions'
    "v4" recipe -- session-level architecture + normal/stealth mixed
    training) in place of build_report()'s per-flow MLP, because that's the
    best-measured detector so far and the Overview dashboard should show its
    alerts/raw data, not an older, weaker model's (tasks/dashboard-data-refresh
    flagged this gap: "versions.py saves comparison and model artifacts but
    no dataset Parquet, version-specific alerts or binary v4 detector/report").

    Output shape matches build_report()'s report.json exactly -- same keys,
    same REPORT_VERSION, same validator -- because a session-level model
    predicts one label per *session*, not per flow, so every flow in a
    session inherits that session's predicted_scenario/confidence, and a
    derived model_score = 1 - P(benign) stands in for the separate binary
    head v1/v2 trained directly. binary_model and multiclass_model are
    therefore two views of the *same* trained GRU, not two separate models;
    both report that GRU's own train_config/history, truthfully.
    """
    out_dir = Path(out_dir)
    sequence_config = sequence_config or SequenceTrainConfig(seed=seed, min_epochs=45)
    class_names = list(SCENARIO_CLASSES)

    normal_config = GenerationConfig(
        seed=seed, num_hosts=num_hosts, duration_minutes=duration_minutes,
        port_scan_episodes=episodes_per_family, auth_bruteforce_episodes=episodes_per_family,
        beaconing_episodes=episodes_per_family, service_flood_episodes=episodes_per_family,
        outbound_transfer_episodes=episodes_per_family, stealth_mode=False,
    )
    stealth_config = dataclasses.replace(normal_config, seed=seed + 1000, stealth_mode=True)

    normal_df, normal_manifest = generate_dataset(normal_config)
    stealth_df, stealth_manifest = generate_dataset(stealth_config)
    # combine_runs, not pd.concat: normal_df/stealth_df are two
    # separately-seeded generation runs whose session_id naming doesn't
    # depend on seed, so a plain concat would collide session_ids between
    # them (confirmed empirically: ~94% collision at this scale) -- this
    # dataset gets grouped by session_id below (build_session_sequences,
    # alert grouping), so it must be collision-free.
    full_df = combine_runs(normal_df, stealth_df)

    combined_manifest = {
        "schema_version": normal_manifest["schema_version"],
        "config": normal_manifest["config"],
        "row_count": normal_manifest["row_count"] + stealth_manifest["row_count"],
        "host_count": normal_manifest["host_count"],
        "scenario_counts": dict(Counter(normal_manifest["scenario_counts"]) + Counter(stealth_manifest["scenario_counts"])),
        "label_counts": dict(Counter(normal_manifest["label_counts"]) + Counter(stealth_manifest["label_counts"])),
        "stealth_config": stealth_manifest["config"],
        "stealth_included": True,
    }
    save_dataset(full_df, combined_manifest, out_dir / "dataset")

    normal_train, normal_val, normal_test = split_dataset(normal_df, val_frac=0.15, test_frac=0.15, seed=seed)
    stealth_train, stealth_val, stealth_test = split_dataset(
        stealth_df, val_frac=0.15, test_frac=0.15, seed=seed + 1000
    )
    # combine_runs, not pd.concat: see full_df above -- the same session_id
    # collision risk applies to every normal+stealth combination point.
    mixed_train = combine_runs(normal_train, stealth_train)
    mixed_val = combine_runs(normal_val, stealth_val)
    test_df = combine_runs(normal_test, stealth_test)

    preprocessor = build_preprocessor(mixed_train, timing_mode=timing_mode)

    # Same real-data-driven sizing as tasks/model-versions' v4 fix (not a
    # fresh guess) -- see max_session_length's docstring.
    if max_seq_len is None:
        max_seq_len = int(max_session_length(mixed_train, mixed_val, test_df) * 1.1) + 1

    train_seq = build_session_sequences(mixed_train, preprocessor, class_names, max_len=max_seq_len)
    val_seq = build_session_sequences(mixed_val, preprocessor, class_names, max_len=max_seq_len)
    test_seq = build_session_sequences(test_df, preprocessor, class_names, max_len=max_seq_len)

    model, history = train_sequence_model(
        train_seq[0], train_seq[1], train_seq[2],
        val_seq[0], val_seq[1], val_seq[2],
        sequence_config, len(class_names),
    )

    probs = predict_sequence_proba(model, test_seq[0], test_seq[1])  # (n_sessions, n_classes)
    session_ids = test_seq[3]
    pred_idx = probs.argmax(axis=-1)
    benign_idx = class_names.index("benign")

    session_pred_label = {sid: class_names[pred_idx[i]] for i, sid in enumerate(session_ids)}
    session_pred_conf = {sid: float(probs[i, pred_idx[i]]) for i, sid in enumerate(session_ids)}
    session_score = {sid: float(1.0 - probs[i, benign_idx]) for i, sid in enumerate(session_ids)}
    session_probs = {sid: probs[i] for i, sid in enumerate(session_ids)}

    scored = test_df.copy()
    scored["predicted_scenario"] = scored["session_id"].map(session_pred_label)
    scored["predicted_scenario_confidence"] = scored["session_id"].map(session_pred_conf).astype(np.float32)
    scored["model_score"] = scored["session_id"].map(session_score).astype(np.float32)
    scored["model_pred"] = (scored["predicted_scenario"] != "benign").astype(np.int64)
    scored = rule_score(scored)

    y_test_bin = extract_labels(test_df)
    y_test_mc = extract_scenario_labels(test_df)
    name_to_int = {name: i for i, name in enumerate(class_names)}
    model_pred_mc = scored["predicted_scenario"].map(name_to_int).to_numpy()
    broadcast_probs = np.stack([session_probs[sid] for sid in scored["session_id"]])

    binary_metrics = evaluate_predictions(y_test_bin, scored["model_pred"].to_numpy())
    rule_metrics = evaluate_predictions(y_test_bin, scored["rule_pred"].to_numpy())
    multiclass_metrics = multiclass_metrics_from_predictions(model_pred_mc, y_test_mc, class_names)
    calibration = expected_calibration_error(broadcast_probs, y_test_mc)

    from ids_ml.features import save_preprocessor  # local import avoids cycle at module load, same as models.train

    artifact_dir = out_dir / "artifact_sequence"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    save_preprocessor(preprocessor, artifact_dir / "preprocessor.pkl")
    (artifact_dir / "metadata.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "architecture": "GRU (session-level)",
        "model_version": "v4",
        "timing_mode": timing_mode,
        "input_dim": train_seq[0].shape[-1],
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "hidden_dim": sequence_config.hidden_dim,
        "max_seq_len": max_seq_len,
        "class_names": class_names,
        "train_config": dataclasses.asdict(sequence_config),
    }, indent=2))

    run_id = f"session-{seed}"
    conn = init_db(out_dir / "alerts.db")
    alerts = generate_alerts_from_scores(scored, run_id, threshold=alert_threshold)
    insert_alerts(conn, alerts)
    group_into_incidents(conn, run_id)
    all_alerts = list_alerts(conn, order_by="start_time")
    export_alerts(conn, out_dir / "alerts.csv", fmt="csv")
    severity_counts = all_alerts["severity"].value_counts().to_dict() if not all_alerts.empty else {}

    train_config_dict = dataclasses.asdict(sequence_config)
    report = {
        "report_version": REPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": {
            "manifest": combined_manifest,
            "split_sizes": {"train": len(mixed_train), "val": len(mixed_val), "test": len(test_df)},
            "sample_flows": full_df[list(_DISPLAY_COLUMNS)].head(sample_size).to_dict(orient="records"),
        },
        "binary_model": {
            "train_config": train_config_dict,
            "history": history,
            "test_metrics": binary_metrics,
            "rule_baseline_metrics": rule_metrics,
        },
        "multiclass_model": {
            "class_names": class_names,
            "train_config": train_config_dict,
            "history": history,
            "test_metrics": multiclass_metrics,
            "calibration": calibration,
        },
        "alerts": {
            "total": int(len(all_alerts)),
            "by_severity": {str(k): int(v) for k, v in severity_counts.items()},
            "sample": all_alerts.head(sample_size).to_dict(orient="records"),
        },
        "artifact_paths": {
            "dataset": str(out_dir / "dataset"),
            "sequence_model": str(artifact_dir),
            "alerts_db": str(out_dir / "alerts.db"),
            "alerts_csv": str(out_dir / "alerts.csv"),
        },
        "detector": {
            "kind": "session_sequence",
            "model_version": "v4",
            "architecture": "GRU (session-level)",
            "training_recipe": "normal+stealth mixed",
            "timing_mode": timing_mode,
            "inference_context": "complete closed sessions",
            "input_dim": train_seq[0].shape[-1],
            "hidden_dim": sequence_config.hidden_dim,
            "max_seq_len": max_seq_len,
            "note": (
                "binary_model and multiclass_model both describe the same "
                "session-level GRU: it predicts one attack-type label per "
                "session, and every flow in that session inherits the "
                "prediction. model_score is 1 - P(benign) from that same "
                "softmax, standing in for a separate binary head."
            ),
        },
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
