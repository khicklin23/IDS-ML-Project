"""Contract test for the unified report.json a UI will be built against.

This intentionally checks *shape* (field names, types, presence) rather than
model quality -- quality thresholds live in test_baseline.py / test_multiclass.py.
If this test needs to change, a UI depending on report.json probably breaks too.
"""

from ids_ml.data.generator import GenerationConfig
from ids_ml.models.train import TrainConfig
from ids_ml.report import REPORT_VERSION, build_report

_CONFIG = GenerationConfig(
    seed=99, num_hosts=40, duration_minutes=90, background_rate_per_min=60.0,
    port_scan_episodes=6, auth_bruteforce_episodes=6, beaconing_episodes=6,
    service_flood_episodes=6, outbound_transfer_episodes=6,
)
# Keep training fast for a shape test; quality thresholds live elsewhere.
_FAST_TRAIN = TrainConfig(seed=99, max_epochs=5, patience=3)


def test_report_has_the_documented_top_level_shape(tmp_path):
    report = build_report(
        99, tmp_path, gen_config=_CONFIG, sample_size=10,
        binary_train_config=_FAST_TRAIN, multiclass_train_config=_FAST_TRAIN,
    )

    assert report["report_version"] == REPORT_VERSION
    assert report["schema_version"]
    assert report["run_id"] == "run-99"

    dataset = report["dataset"]
    assert dataset["manifest"]["row_count"] > 0
    assert set(dataset["split_sizes"]) == {"train", "val", "test"}
    assert len(dataset["sample_flows"]) == 10
    sample = dataset["sample_flows"][0]
    for field in ("flow_id", "event_time", "src_ip", "dst_ip", "protocol", "scenario", "label"):
        assert field in sample

    binary = report["binary_model"]
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "confusion_matrix", "n"):
        assert key in binary["test_metrics"]
        assert key in binary["rule_baseline_metrics"]

    multiclass = report["multiclass_model"]
    assert multiclass["class_names"]
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix", "n"):
        assert key in multiclass["test_metrics"]
    for class_name in multiclass["class_names"]:
        assert class_name in multiclass["test_metrics"]["per_class"]
        for stat in ("precision", "recall", "f1", "support"):
            assert stat in multiclass["test_metrics"]["per_class"][class_name]
    assert "ece" in multiclass["calibration"]
    assert "bins" in multiclass["calibration"]

    alerts = report["alerts"]
    assert alerts["total"] >= 0
    assert isinstance(alerts["by_severity"], dict)
    if alerts["sample"]:
        alert = alerts["sample"][0]
        for field in (
            "alert_id", "src_ip", "dst_ip", "severity", "scenario_ground_truth",
            "predicted_scenario", "predicted_scenario_confidence", "incident_id",
        ):
            assert field in alert

    paths = report["artifact_paths"]
    for key in ("dataset", "binary_model", "multiclass_model", "alerts_db", "alerts_csv"):
        assert key in paths

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "dataset" / "flows.parquet").exists()
    assert (tmp_path / "artifact_binary" / "model.pt").exists()
    assert (tmp_path / "artifact_multiclass" / "model.pt").exists()
    assert (tmp_path / "alerts.csv").exists()
