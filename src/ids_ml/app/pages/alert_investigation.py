"""Alert investigation: filter, drill in, acknowledge/annotate, export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ids_ml.alerts import export_alerts, list_alerts, update_alert
from ids_ml.app import state, theme
from ids_ml.app.components import require_scores

_STATUSES = ["new", "acknowledged", "closed"]
_SEVERITIES = ["critical", "high", "medium", "low"]


def render() -> None:
    st.title("Alert investigation")
    st.caption("Alerts are generated from the model's scores only -- the rule baseline never decides severity here.")

    if not require_scores():
        return

    conn = state.get_alert_conn()

    c1, c2 = st.columns(2)
    severity_filter = c1.multiselect("Severity", _SEVERITIES, default=_SEVERITIES)
    status_filter = c2.selectbox("Status", ["(all)"] + _STATUSES)

    all_alerts = list_alerts(conn, order_by="start_time", ascending=False)
    if all_alerts.empty:
        st.info("No alerts at the current threshold. Lower it on Detection & Replay to see some.")
        return

    severity_counts = all_alerts["severity"].value_counts()
    fig = go.Figure(
        go.Bar(
            x=_SEVERITIES,
            y=[int(severity_counts.get(s, 0)) for s in _SEVERITIES],
            marker_color=[theme.SEVERITY_COLOR[s] for s in _SEVERITIES],
        )
    )
    fig.update_layout(xaxis_title="Severity", yaxis_title="Alerts")
    theme.apply_layout(fig)
    st.plotly_chart(fig, width="stretch")

    filtered = all_alerts[all_alerts["severity"].isin(severity_filter)]
    if status_filter != "(all)":
        filtered = filtered[filtered["status"] == status_filter]

    st.dataframe(
        filtered[["alert_id", "severity", "status", "src_host_id", "dst_host_id", "start_time", "flow_count", "incident_id"]],
        width="stretch", hide_index=True,
    )

    if filtered.empty:
        st.caption("No alerts match the current filters.")
        return

    st.divider()
    st.subheader("Drill down")
    selected_id = st.selectbox("Alert", filtered["alert_id"].tolist())
    alert = filtered.set_index("alert_id").loc[selected_id]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(theme.ground_truth_badge("Ground-truth scenario") + f": **{alert['scenario_ground_truth']}**", unsafe_allow_html=True)
        st.caption(
            "This is the generator's known-correct label for evaluation -- the "
            "current binary model does not identify attack type itself yet."
        )
        st.write(f"Hosts: `{alert['src_host_id']}` -> `{alert['dst_host_id']}`")
        st.write(f"Window: {alert['start_time']:.1f}s -- {alert['end_time']:.1f}s ({alert['flow_count']} flows)")
    with c2:
        st.markdown(theme.inference_badge("Model scores") , unsafe_allow_html=True)
        st.write(f"Max score: {alert['max_score']:.3f}  ·  Mean score: {alert['mean_score']:.3f}")
        st.write(f"Severity: **{alert['severity']}**  ·  Incident: `{alert['incident_id']}`")

    with st.form(f"ack_form_{selected_id}"):
        new_status = st.selectbox("Status", _STATUSES, index=_STATUSES.index(alert["status"]))
        notes = st.text_area("Notes", value=alert["notes"] or "")
        if st.form_submit_button("Save"):
            update_alert(conn, selected_id, status=new_status, notes=notes)
            st.success("Saved.")
            st.rerun()

    related = all_alerts[all_alerts["incident_id"] == alert["incident_id"]]
    if len(related) > 1:
        st.caption(f"{len(related)} alerts grouped into this incident (same host, nearby in time):")
        st.dataframe(
            related[["alert_id", "severity", "scenario_ground_truth", "start_time"]],
            width="stretch", hide_index=True,
        )

    st.divider()
    st.subheader("Export")
    fmt = st.radio("Format", ["csv", "json"], horizontal=True)
    if st.button("Prepare export"):
        tmp_path = Path(tempfile.mkdtemp(prefix="ids_ml_alerts_")) / f"alerts.{fmt}"
        export_alerts(conn, tmp_path, fmt=fmt)
        st.session_state["_alert_export_path"] = tmp_path
    export_path = st.session_state.get("_alert_export_path")
    if export_path and Path(export_path).exists():
        st.download_button(f"Download {Path(export_path).name}", data=Path(export_path).read_bytes(), file_name=Path(export_path).name)
