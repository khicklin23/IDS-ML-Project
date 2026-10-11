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
from pathlib import Path

from ids_ml.alerts import export_alerts, generate_alerts_from_scores, group_into_incidents, init_db, insert_alerts, list_alerts
from ids_ml.baseline import rule_score
from ids_ml.data.generator import GenerationConfig, generate_dataset, save_dataset, split_dataset
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
    predict_class_proba,
)
from ids_ml.models.train import TrainConfig, save_artifact, train_model, train_multiclass_model

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
