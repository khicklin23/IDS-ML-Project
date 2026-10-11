"""Scenario and data lab: configure and generate a synthetic bundle."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ids_ml.app import state, theme
from ids_ml.data.generator import GenerationConfig, save_dataset

# Rough midpoints used only for the pre-generation size estimate -- actual
# counts depend on random draws and are shown for real after generation.
_EPISODE_ROW_ESTIMATE = {
    "port_scan_episodes": 115,
    "auth_bruteforce_episodes": 90,
    "beaconing_episodes": 10,
    "service_flood_episodes": 175,
    "outbound_transfer_episodes": 12,
}


def _estimate_rows(config: GenerationConfig) -> int:
    background = config.background_rate_per_min * config.duration_minutes
    episodes = sum(
        getattr(config, field, 0) * per_episode
        for field, per_episode in _EPISODE_ROW_ESTIMATE.items()
    )
    return int(background + episodes)


def render() -> None:
    st.title("Scenario & data lab")
    st.caption("Configure a seeded synthetic environment, preview its size, then generate it.")

    current = st.session_state.gen_config

    with st.form("gen_config_form"):
        c1, c2 = st.columns(2)
        seed = c1.number_input("Seed", value=current.seed, step=1)
        num_hosts = c2.number_input("Hosts", min_value=4, max_value=500, value=current.num_hosts, step=1)

        c1, c2 = st.columns(2)
        duration_minutes = c1.number_input(
            "Duration (minutes)", min_value=1.0, max_value=24 * 60.0,
            value=float(current.duration_minutes), step=10.0,
        )
        background_rate = c2.number_input(
            "Background rate (flows/min)", min_value=0.0, max_value=2000.0,
            value=float(current.background_rate_per_min), step=10.0,
        )

        st.markdown("**Attack scenario episodes** (ground-truth attack families injected into the bundle)")
        c1, c2, c3 = st.columns(3)
        port_scan = c1.number_input("Port scan", min_value=0, max_value=50, value=current.port_scan_episodes)
        auth_bf = c2.number_input("Auth brute-force", min_value=0, max_value=50, value=current.auth_bruteforce_episodes)
        beaconing = c3.number_input("Beaconing", min_value=0, max_value=50, value=current.beaconing_episodes)

        c1, c2, c3 = st.columns(3)
        service_flood = c1.number_input(
            "Service flood", min_value=0, max_value=50,
            value=getattr(current, "service_flood_episodes", 0),
            help="Opt-in family added by the scenario-library task.",
        )
        outbound_transfer = c2.number_input(
            "Outbound transfer", min_value=0, max_value=50,
            value=getattr(current, "outbound_transfer_episodes", 0),
            help="Opt-in family added by the scenario-library task.",
        )
        realistic_background = c3.checkbox(
            "Realistic background counters",
            value=getattr(current, "realistic_background", False),
            help="Service-aware packet/byte/timing distributions for benign traffic.",
        )

        submitted = st.form_submit_button("Preview", width="stretch")

    field_names = {f.name for f in dataclasses.fields(GenerationConfig)}
    candidate_kwargs = dict(
        seed=int(seed),
        num_hosts=int(num_hosts),
        duration_minutes=float(duration_minutes),
        background_rate_per_min=float(background_rate),
        port_scan_episodes=int(port_scan),
        auth_bruteforce_episodes=int(auth_bf),
        beaconing_episodes=int(beaconing),
    )
    if "service_flood_episodes" in field_names:
        candidate_kwargs["service_flood_episodes"] = int(service_flood)
    if "outbound_transfer_episodes" in field_names:
        candidate_kwargs["outbound_transfer_episodes"] = int(outbound_transfer)
    if "realistic_background" in field_names:
        candidate_kwargs["realistic_background"] = bool(realistic_background)

    candidate = GenerationConfig(**candidate_kwargs)

    st.subheader("Preview")
    st.caption("Estimated size only -- episode lengths vary with random draws; exact counts appear after generation.")
    estimate = _estimate_rows(candidate)
    st.metric("Estimated flows", f"~{estimate:,}")

    if st.button("Generate dataset", type="primary"):
        with st.spinner("Generating..."):
            state.generate(candidate)
        st.success(f"Generated {st.session_state.dataset[1]['row_count']:,} flows.")
        if st.session_state.split_warning:
            st.warning(st.session_state.split_warning)

    if state.has_dataset():
        st.divider()
        st.subheader("Last generated bundle")
        df, manifest = st.session_state.dataset
        st.markdown(theme.ground_truth_badge(), unsafe_allow_html=True)

        fig = go.Figure(
            go.Bar(
                x=list(manifest["scenario_counts"].keys()),
                y=list(manifest["scenario_counts"].values()),
                marker_color=theme.TEAL,
            )
        )
        fig.update_layout(xaxis_title="Scenario", yaxis_title="Flows")
        theme.apply_layout(fig)
        st.plotly_chart(fig, width="stretch")

        with st.expander("Export dataset + manifest"):
            if st.button("Prepare export"):
                tmp_dir = Path(tempfile.mkdtemp(prefix="ids_ml_export_"))
                paths = save_dataset(df, manifest, tmp_dir)
                st.session_state["_export_paths"] = paths
            paths = st.session_state.get("_export_paths")
            if paths:
                st.download_button(
                    "Download flows.parquet", data=Path(paths["data"]).read_bytes(),
                    file_name="flows.parquet",
                )
                st.download_button(
                    "Download manifest.json", data=Path(paths["manifest"]).read_bytes(),
                    file_name="manifest.json",
                )
