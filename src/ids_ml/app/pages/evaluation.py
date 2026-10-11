"""Evaluation: confusion matrix, per-class metrics, PR curve, model vs rule."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from ids_ml.app import state, theme
from ids_ml.app.components import metric_row, require_scores, synthetic_disclaimer
from ids_ml.features import extract_labels


def _confusion_figure(cm: dict, *, title: str) -> go.Figure:
    z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted suspicious", "Predicted normal"],
            y=["Actual suspicious", "Actual normal"],
            colorscale=[[0, theme.SURFACE], [1, theme.BLUE]],
            showscale=False,
            text=z,
            texttemplate="%{text}",
            textfont=dict(color=theme.TEXT_PRIMARY),
        )
    )
    fig.update_layout(title=title)
    theme.apply_layout(fig, height=320)
    return fig


def render() -> None:
    st.title("Evaluation")
    st.caption("Held-out test split only. Scenario provenance was excluded from model inputs during training.")
    synthetic_disclaimer()

    if not require_scores():
        return

    artifact = st.session_state.artifact
    scored = st.session_state.scored_test

    st.markdown(theme.inference_badge() + " &nbsp; " + theme.rule_badge(), unsafe_allow_html=True)
    metric_row(artifact["test_metrics"], label="Model")
    metric_row(artifact["rule_metrics"], label="Rule")

    c1, c2 = st.columns(2)
    c1.plotly_chart(_confusion_figure(artifact["test_metrics"]["confusion_matrix"], title="Model"), width="stretch")
    c2.plotly_chart(_confusion_figure(artifact["rule_metrics"]["confusion_matrix"], title="Rule baseline"), width="stretch")

    st.divider()
    st.subheader("Precision-recall curve (model)")
    st.markdown(theme.inference_badge(), unsafe_allow_html=True)
    _, _, test_df = st.session_state.splits
    y_test = extract_labels(test_df)
    precision, recall, thresholds = precision_recall_curve(y_test, scored["model_score"])

    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color=theme.BLUE, width=2)))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1])
    theme.apply_layout(fig)
    current_threshold = st.session_state.threshold
    idx = (thresholds <= current_threshold).sum() - 1
    if 0 <= idx < len(precision) - 1:
        fig.add_trace(
            go.Scatter(
                x=[recall[idx]], y=[precision[idx]], mode="markers",
                marker=dict(size=12, color=theme.RED), name=f"current threshold {current_threshold:.2f}",
            )
        )
    st.plotly_chart(fig, width="stretch")
    st.caption("Red marker: precision/recall at the threshold currently set on Detection & Replay.")
