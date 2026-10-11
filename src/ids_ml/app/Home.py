"""Entry point: `streamlit run src/ids_ml/app/Home.py`."""

from __future__ import annotations

import streamlit as st

from ids_ml.app import state, theme
from ids_ml.app.pages import (
    alert_investigation,
    data_lab,
    detection_replay,
    evaluation,
    overview,
    training_studio,
)

st.set_page_config(page_title="IDS Workbench", page_icon="\U0001F6F0", layout="wide")
st.markdown(theme.inject_css(), unsafe_allow_html=True)
state.init_session_state()

# st.Page's `icon` takes a literal emoji character or a `:material/...:`
# shortcode -- NOT an emoji :shortcode: like GitHub/Slack use. Literal
# emoji below, not shortcodes.
pages = [
    st.Page(overview.render, title="Operations Overview", icon="\U0001F6F0", url_path="overview", default=True),
    st.Page(data_lab.render, title="Scenario & Data Lab", icon="\U0001F9EA", url_path="data-lab"),
    st.Page(training_studio.render, title="Training Studio", icon="\U0001F9E0", url_path="training-studio"),
    st.Page(detection_replay.render, title="Detection & Replay", icon="▶", url_path="detection-replay"),
    st.Page(alert_investigation.render, title="Alert Investigation", icon="\U0001F6A8", url_path="alert-investigation"),
    st.Page(evaluation.render, title="Evaluation", icon="\U0001F4CA", url_path="evaluation"),
]

st.navigation(pages).run()
