"""Alert generation and investigation storage.

Separate module from inference/replay and from the UI, per
docs/PRODUCT_VISION.md's component-separation requirement. Alerts are
generated from the model's scores (ids_ml.inference.score_dataset) --
the model is the primary detector; this module only aggregates its output
into investigable units. It does not re-decide severity from rule logic.

Storage is plain SQLite (stdlib, no new dependency) so alerts persist
across a UI session without pulling in a database server for an
educational project.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

_SEVERITY_THRESHOLDS = (
    (0.95, "critical"),
    (0.80, "high"),
    (0.60, "medium"),
)
_DEFAULT_SEVERITY = "low"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    src_host_id TEXT NOT NULL,
    dst_host_id TEXT NOT NULL,
    src_ip TEXT NOT NULL,
    dst_ip TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    flow_count INTEGER NOT NULL,
    max_score REAL NOT NULL,
    mean_score REAL NOT NULL,
    severity TEXT NOT NULL,
    scenario_ground_truth TEXT NOT NULL,
    predicted_scenario TEXT,
    predicted_scenario_confidence REAL,
    status TEXT NOT NULL DEFAULT 'new',
    notes TEXT,
    incident_id TEXT,
    created_at TEXT NOT NULL
)
"""


def init_db(path: Path | str, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """check_same_thread=False is needed when the connection is kept across
    Streamlit reruns/fragments, which may execute on different worker
    threads sequentially (never concurrently) within one session."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _severity_for(max_score: float) -> str:
    for cutoff, label in _SEVERITY_THRESHOLDS:
        if max_score >= cutoff:
            return label
    return _DEFAULT_SEVERITY


def generate_alerts_from_scores(
    scored_df: pd.DataFrame,
    run_id: str,
    *,
    threshold: float = 0.5,
) -> list[dict]:
    """Group model-scored flows by session_id into alert records.

    `scenario_ground_truth` is the generator's known-correct label for this
    synthetic episode -- it exists for evaluation and UI labeling only, and
    must be presented as ground truth, never as something the detector
    "figured out". If scored_df also carries `predicted_scenario`/
    `predicted_scenario_confidence` (ids_ml.inference.score_dataset_multiclass
    was run on the same data), those are included as the model's own
    attack-type guess -- kept as separate, clearly-named columns so a UI
    never confuses the two. Omitted (NULL) when multiclass scoring wasn't run.
    """
    alerts: list[dict] = []
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    has_predicted_type = "predicted_scenario" in scored_df.columns

    flagged_sessions = scored_df.loc[scored_df["model_pred"] == 1, "session_id"].unique()
    for i, session_id in enumerate(sorted(flagged_sessions)):
        group = scored_df[scored_df["session_id"] == session_id]
        max_score = float(group["model_score"].max())
        alert = {
            "alert_id": f"{run_id}-alert-{i:06d}",
            "run_id": run_id,
            "session_id": str(session_id),
            "src_host_id": str(group["src_host_id"].iloc[0]),
            "dst_host_id": str(group["dst_host_id"].mode().iloc[0]),
            "src_ip": str(group["src_ip"].iloc[0]),
            "dst_ip": str(group["dst_ip"].mode().iloc[0]),
            "start_time": float(group["event_time"].min()),
            "end_time": float(group["event_time"].max()),
            "flow_count": int(len(group)),
            "max_score": max_score,
            "mean_score": float(group["model_score"].mean()),
            "severity": _severity_for(max_score),
            "scenario_ground_truth": str(group["scenario"].iloc[0]),
            "predicted_scenario": None,
            "predicted_scenario_confidence": None,
            "status": "new",
            "notes": None,
            "incident_id": None,
            "created_at": created_at,
        }
        if has_predicted_type:
            alert["predicted_scenario"] = str(group["predicted_scenario"].mode().iloc[0])
            alert["predicted_scenario_confidence"] = float(group["predicted_scenario_confidence"].mean())
        alerts.append(alert)
    return alerts


def insert_alerts(conn: sqlite3.Connection, alerts: list[dict]) -> int:
    if not alerts:
        return 0
    columns = list(alerts[0].keys())
    placeholders = ", ".join(f":{c}" for c in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO alerts ({', '.join(columns)}) VALUES ({placeholders})",
        alerts,
    )
    conn.commit()
    return len(alerts)


def list_alerts(
    conn: sqlite3.Connection,
    *,
    severity: str | None = None,
    status: str | None = None,
    run_id: str | None = None,
    order_by: str = "start_time",
    ascending: bool = True,
) -> pd.DataFrame:
    allowed_order_cols = {
        "start_time", "end_time", "max_score", "mean_score", "severity", "flow_count", "created_at",
    }
    if order_by not in allowed_order_cols:
        raise ValueError(f"order_by must be one of {sorted(allowed_order_cols)}")

    clauses, params = [], {}
    if severity is not None:
        clauses.append("severity = :severity")
        params["severity"] = severity
    if status is not None:
        clauses.append("status = :status")
        params["status"] = status
    if run_id is not None:
        clauses.append("run_id = :run_id")
        params["run_id"] = run_id

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    direction = "ASC" if ascending else "DESC"
    query = f"SELECT * FROM alerts {where} ORDER BY {order_by} {direction}"
    return pd.read_sql_query(query, conn, params=params)


def update_alert(
    conn: sqlite3.Connection,
    alert_id: str,
    *,
    status: str | None = None,
    notes: str | None = None,
) -> None:
    if status is None and notes is None:
        return
    sets, params = [], {"alert_id": alert_id}
    if status is not None:
        sets.append("status = :status")
        params["status"] = status
    if notes is not None:
        sets.append("notes = :notes")
        params["notes"] = notes

    cursor = conn.execute(f"UPDATE alerts SET {', '.join(sets)} WHERE alert_id = :alert_id", params)
    conn.commit()
    if cursor.rowcount == 0:
        raise KeyError(f"no alert with alert_id={alert_id!r}")


def group_into_incidents(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    time_window_s: float = 300.0,
) -> dict[str, str]:
    """Group alerts sharing a source host within time_window_s of each other
    into one incident. Simple, deterministic, host-centric grouping -- not
    a claim about real attack-chain correlation.
    """
    df = list_alerts(conn, run_id=run_id, order_by="start_time")
    assignment: dict[str, str] = {}
    counter = 0

    for host, group in df.groupby("src_host_id"):
        group = group.sort_values("start_time")
        current_incident = None
        last_end = None
        for _, row in group.iterrows():
            if current_incident is None or row["start_time"] - last_end > time_window_s:
                current_incident = f"{run_id}-incident-{counter:04d}"
                counter += 1
            assignment[row["alert_id"]] = current_incident
            last_end = max(last_end or row["end_time"], row["end_time"])

    conn.executemany(
        "UPDATE alerts SET incident_id = :incident_id WHERE alert_id = :alert_id",
        [{"incident_id": v, "alert_id": k} for k, v in assignment.items()],
    )
    conn.commit()
    return assignment


def export_alerts(conn: sqlite3.Connection, path: Path | str, *, fmt: str = "csv") -> Path:
    path = Path(path)
    df = list_alerts(conn)
    if fmt == "csv":
        df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    elif fmt == "json":
        path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, sort_keys=True))
    else:
        raise ValueError(f"unsupported export format: {fmt!r}")
    return path
