"""Detection & replay: score the model, compare the rule baseline, replay events."""

from __future__ import annotations

import streamlit as st

from ids_ml.app import state, theme
from ids_ml.app.components import metric_row, require_artifact, synthetic_disclaimer

_SPEED_OPTIONS = [10, 50, 200, 1000, 5000]
_DISPLAY_COLUMNS = [
    "event_time", "src_host_id", "dst_host_id", "protocol", "service",
    "scenario", "model_score", "model_pred", "rule_pred",
]


def render() -> None:
    st.title("Detection & replay")
    st.caption(
        "The model is the primary detector. The rule baseline is shown only "
        "for comparison -- it is not the thing deciding severity or alerts."
    )

    if not require_artifact():
        return

    threshold = st.slider(
        "Detection threshold -- P(suspicious) >=", 0.0, 1.0, float(st.session_state.threshold), 0.01
    )
    if st.button("Score held-out set with this threshold", type="primary"):
        with st.spinner("Scoring..."):
            state.score(threshold=threshold)
            state.refresh_alerts()

    if not state.has_scores():
        st.info("Score the held-out set to see detection results and enable replay.")
        return

    artifact = st.session_state.artifact
    st.subheader("Model vs. rule baseline (same held-out set)")
    st.markdown(
        theme.inference_badge() + " &nbsp; " + theme.rule_badge(),
        unsafe_allow_html=True,
    )
    metric_row(artifact["test_metrics"], label="Model")
    metric_row(artifact["rule_metrics"], label="Rule")
    st.caption(
        "The rule baseline is a short/failed/rejected-connection heuristic. It has a "
        "documented blind spot: it cannot see beaconing-style traffic because those "
        "flows look like ordinary short 'established' connections to a simple rule."
    )
    synthetic_disclaimer()

    st.divider()
    st.subheader("Replay")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Play"):
        state.get_replay().play()
    if c2.button("Pause"):
        state.get_replay().pause()
    if c3.button("Reset"):
        state.get_replay().reset()
    speed = c4.selectbox("Speed", _SPEED_OPTIONS, index=2, format_func=lambda s: f"{s}x")
    state.get_replay().set_speed(float(speed))

    seek_col, jump_col = st.columns([3, 1])
    replay = state.get_replay()
    seek_target = seek_col.number_input(
        "Seek to simulated time (s)", min_value=replay.min_time, max_value=replay.max_time,
        value=replay.sim_time, step=10.0,
    )
    if jump_col.button("Jump", width="stretch"):
        replay.seek(seek_target)

    _replay_panel()


@st.fragment(run_every=1.0)
def _replay_panel() -> None:
    replay = state.get_replay()
    if replay.playing:
        replay.advance(1.0)

    span = max(replay.max_time - replay.min_time, 1e-9)
    frac = (replay.sim_time - replay.min_time) / span
    st.progress(min(max(frac, 0.0), 1.0))
    status = "playing" if replay.playing else "paused"
    st.caption(
        f"Simulated time: {replay.sim_time:,.1f}s / {replay.max_time:,.1f}s "
        f"({status}, {replay.speed:g}x) -- this is replay time, not wall-clock time."
    )

    window_s = max(span * 0.02, 5.0)
    recent = replay.window(window_s)
    st.markdown(theme.inference_badge("Model score/pred") + " &nbsp; " + theme.rule_badge("Rule pred"), unsafe_allow_html=True)
    st.dataframe(
        recent[_DISPLAY_COLUMNS].sort_values("event_time", ascending=False).head(25),
        width="stretch",
        hide_index=True,
    )
