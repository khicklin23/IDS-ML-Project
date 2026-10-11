import json

import pandas as pd
import pytest

from ids_ml.alerts import (
    export_alerts,
    generate_alerts_from_scores,
    group_into_incidents,
    init_db,
    insert_alerts,
    list_alerts,
    update_alert,
)

RUN_ID = "run-test"


def _scored_df() -> pd.DataFrame:
    rows = [
        # session s1: port_scan, all flagged, same host as s2, close in time -> one incident
        dict(session_id="s1", src_host_id="host-01", dst_host_id="host-02", event_time=0.0,
             model_score=0.90, model_pred=1, scenario="port_scan"),
        dict(session_id="s1", src_host_id="host-01", dst_host_id="host-02", event_time=1.0,
             model_score=0.95, model_pred=1, scenario="port_scan"),
        dict(session_id="s1", src_host_id="host-01", dst_host_id="host-02", event_time=2.0,
             model_score=0.99, model_pred=1, scenario="port_scan"),
        # session s2: auth_bruteforce, same src host as s1, 248s later -> within 300s window
        dict(session_id="s2", src_host_id="host-01", dst_host_id="host-03", event_time=250.0,
             model_score=0.85, model_pred=1, scenario="auth_bruteforce"),
        dict(session_id="s2", src_host_id="host-01", dst_host_id="host-03", event_time=260.0,
             model_score=0.82, model_pred=1, scenario="auth_bruteforce"),
        # session s3: beaconing, different host, far away in time -> separate incident
        dict(session_id="s3", src_host_id="host-05", dst_host_id="host-09", event_time=1000.0,
             model_score=0.99, model_pred=1, scenario="beaconing"),
        # session s4: benign, never flagged -> no alert
        dict(session_id="s4", src_host_id="host-07", dst_host_id="host-08", event_time=5.0,
             model_score=0.10, model_pred=0, scenario="benign"),
    ]
    df = pd.DataFrame(rows)
    to_ip = lambda host_id: f"10.0.0.{int(host_id.split('-')[1])}"  # noqa: E731
    df["src_ip"] = df["src_host_id"].map(to_ip)
    df["dst_ip"] = df["dst_host_id"].map(to_ip)
    return df


def test_generate_alerts_skips_unflagged_sessions_and_buckets_severity():
    alerts = generate_alerts_from_scores(_scored_df(), RUN_ID)
    session_ids = {a["session_id"] for a in alerts}
    assert session_ids == {"s1", "s2", "s3"}

    by_session = {a["session_id"]: a for a in alerts}
    assert by_session["s1"]["severity"] == "critical"  # max_score 0.99
    assert by_session["s2"]["severity"] == "high"  # max_score 0.85
    assert by_session["s1"]["flow_count"] == 3
    assert by_session["s1"]["scenario_ground_truth"] == "port_scan"
    assert by_session["s1"]["src_ip"] == "10.0.0.1"
    assert by_session["s1"]["predicted_scenario"] is None  # no multiclass scoring on this df


def test_generate_alerts_includes_predicted_scenario_when_present():
    df = _scored_df()
    df["predicted_scenario"] = df["scenario"]  # pretend the model nailed it, for this test
    df["predicted_scenario_confidence"] = 0.8

    alerts = generate_alerts_from_scores(df, RUN_ID)
    by_session = {a["session_id"]: a for a in alerts}
    assert by_session["s1"]["predicted_scenario"] == "port_scan"
    assert by_session["s1"]["predicted_scenario_confidence"] == pytest.approx(0.8)


def test_insert_and_list_round_trip_with_filters():
    conn = init_db(":memory:")
    alerts = generate_alerts_from_scores(_scored_df(), RUN_ID)
    inserted = insert_alerts(conn, alerts)
    assert inserted == 3

    all_alerts = list_alerts(conn)
    assert len(all_alerts) == 3

    critical = list_alerts(conn, severity="critical")
    assert set(critical["session_id"]) == {"s1", "s3"}

    by_run = list_alerts(conn, run_id=RUN_ID)
    assert len(by_run) == 3
    assert len(list_alerts(conn, run_id="other-run")) == 0


def test_update_alert_ack_and_annotate_and_missing_id():
    conn = init_db(":memory:")
    alerts = generate_alerts_from_scores(_scored_df(), RUN_ID)
    insert_alerts(conn, alerts)
    alert_id = alerts[0]["alert_id"]

    update_alert(conn, alert_id, status="acknowledged", notes="looks like a scan")
    row = list_alerts(conn).set_index("alert_id").loc[alert_id]
    assert row["status"] == "acknowledged"
    assert row["notes"] == "looks like a scan"

    with pytest.raises(KeyError):
        update_alert(conn, "does-not-exist", status="closed")


def test_group_into_incidents_groups_by_host_and_time_window():
    conn = init_db(":memory:")
    alerts = generate_alerts_from_scores(_scored_df(), RUN_ID)
    insert_alerts(conn, alerts)

    mapping = group_into_incidents(conn, RUN_ID, time_window_s=300.0)
    by_session = {a["session_id"]: a["alert_id"] for a in alerts}

    assert mapping[by_session["s1"]] == mapping[by_session["s2"]]  # same host, within window
    assert mapping[by_session["s3"]] != mapping[by_session["s1"]]  # different host

    persisted = list_alerts(conn).set_index("session_id")
    assert persisted.loc["s1", "incident_id"] == persisted.loc["s2", "incident_id"]


def test_export_alerts_csv_and_json(tmp_path):
    conn = init_db(":memory:")
    insert_alerts(conn, generate_alerts_from_scores(_scored_df(), RUN_ID))

    csv_path = export_alerts(conn, tmp_path / "alerts.csv", fmt="csv")
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 3

    json_path = export_alerts(conn, tmp_path / "alerts.json", fmt="json")
    assert len(json.loads(json_path.read_text())) == 3

    with pytest.raises(ValueError):
        export_alerts(conn, tmp_path / "alerts.xyz", fmt="xyz")
