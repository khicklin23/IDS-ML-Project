"""Operations overview: a glance at the current simulated environment."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ids_ml.app import state, theme
from ids_ml.app.components import require_dataset, synthetic_disclaimer


def render() -> None:
    st.title("Operations overview")
    st.caption("Simulated sensor status for the currently generated scenario bundle.")

    if not require_dataset():
        return

    df, manifest = st.session_state.dataset
    config = st.session_state.gen_config

    st.markdown(theme.ground_truth_badge(), unsafe_allow_html=True)
    st.caption(
        "Counts below come from the generator's own labels -- they describe what "
        "was simulated, not what any detector concluded."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hosts", manifest["host_count"])
    c2.metric("Flows", f"{manifest['row_count']:,}")
    c3.metric("Simulated duration", f"{config.duration_minutes:.0f} min")
    c4.metric(
        "Suspicious share",
        f"{manifest['label_counts'].get('suspicious', 0) / manifest['row_count']:.1%}",
    )

    if state.has_scores():
        conn = state.get_alert_conn()
        alerts = pd.read_sql_query("SELECT severity FROM alerts", conn)
        counts = alerts["severity"].value_counts()
        a1, a2, a3, a4 = st.columns(4)
        for col, sev in zip((a1, a2, a3, a4), ("critical", "high", "medium", "low")):
            col.metric(f"{sev.capitalize()} alerts", int(counts.get(sev, 0)))
    else:
        st.caption("Score the held-out set on **Detection & Replay** to see alert counts here.")

    st.divider()
    st.subheader("Event timeline")
    st.caption(
        "Simulated event time on the x-axis -- this is replay time, not wall-clock "
        "time. One bar per time bucket, colored by ground-truth scenario."
    )

    bucket_s = max(config.duration_s / 60.0, 1.0)
    timeline = df.copy()
    timeline["bucket"] = (timeline["event_time"] // bucket_s) * bucket_s

    fig = go.Figure()
    scenario_colors = {
        "benign": theme.NORMAL_COLOR,
        "port_scan": theme.AMBER,
        "auth_bruteforce": theme.ORANGE,
        "beaconing": theme.MAGENTA,
        "service_flood": theme.RED,
        "outbound_transfer": theme.VIOLET,
    }
    for scenario_name, group in timeline.groupby("scenario", observed=True):
        counts = group.groupby("bucket").size()
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts.values,
                name=str(scenario_name),
                marker_color=scenario_colors.get(str(scenario_name), theme.TEXT_MUTED),
            )
        )
    fig.update_layout(barmode="stack", xaxis_title="Simulated time (s)", yaxis_title="Flows")
    theme.apply_layout(fig)
    st.plotly_chart(fig, width="stretch")

    synthetic_disclaimer()
