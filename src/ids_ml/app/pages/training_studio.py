"""Training studio: configure, train, and watch the baseline classifier."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ids_ml.app import state, theme
from ids_ml.app.components import metric_row, require_dataset, synthetic_disclaimer
from ids_ml.models.train import TrainConfig


def _schematic_figure(layer_sizes: list[int]) -> go.Figure:
    fig = go.Figure()
    n_layers = len(layer_sizes)
    max_nodes = min(max(layer_sizes), 8)

    positions = {}
    for li, size in enumerate(layer_sizes):
        shown = min(size, max_nodes)
        for ni in range(shown):
            y = (ni - (shown - 1) / 2.0)
            positions[(li, ni)] = (li, y)

    for li in range(n_layers - 1):
        left_n = min(layer_sizes[li], max_nodes)
        right_n = min(layer_sizes[li + 1], max_nodes)
        xs, ys = [], []
        for a in range(left_n):
            for b in range(right_n):
                x0, y0 = positions[(li, a)]
                x1, y1 = positions[(li + 1, b)]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=theme.GRIDLINE, width=1), hoverinfo="skip", showlegend=False))

    for li, size in enumerate(layer_sizes):
        shown = min(size, max_nodes)
        xs = [positions[(li, n)][0] for n in range(shown)]
        ys = [positions[(li, n)][1] for n in range(shown)]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers", marker=dict(size=14, color=theme.BLUE, line=dict(color=theme.SURFACE, width=2)),
                showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_annotation(x=li, y=max_nodes / 2 + 1, text=f"{size}", showarrow=False, font=dict(color=theme.TEXT_SECONDARY))

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=220, showlegend=False,
    )
    theme.apply_layout(fig, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render() -> None:
    st.title("Training studio")
    st.caption("Small MLP baseline -- intentionally simple, to compare future complexity against.")

    if not require_dataset():
        return

    current = st.session_state.train_config
    with st.form("train_config_form"):
        c1, c2, c3 = st.columns(3)
        hidden_dim = c1.number_input("Hidden dim", min_value=4, max_value=256, value=current.hidden_dim)
        dropout = c2.slider("Dropout", 0.0, 0.5, float(current.dropout))
        lr = c3.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=float(current.lr), format="%.5f")

        c1, c2, c3 = st.columns(3)
        max_epochs = c1.number_input("Max epochs", min_value=1, max_value=500, value=current.max_epochs)
        patience = c2.number_input("Early-stop patience", min_value=1, max_value=100, value=current.patience)
        batch_size = c3.number_input("Batch size", min_value=4, max_value=2048, value=current.batch_size)

        class_weight = st.checkbox("Class-weight loss for imbalance", value=current.class_weight)
        submitted = st.form_submit_button("Train", type="primary", width="stretch")

    if submitted:
        train_config = TrainConfig(
            seed=current.seed,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            lr=float(lr),
            max_epochs=int(max_epochs),
            patience=int(patience),
            batch_size=int(batch_size),
            class_weight=bool(class_weight),
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        seen: list[dict] = []

        def on_epoch(entry: dict) -> None:
            seen.append(entry)
            progress_bar.progress(min((entry["epoch"] + 1) / train_config.max_epochs, 1.0))
            status_text.text(
                f"Epoch {entry['epoch'] + 1}/{train_config.max_epochs} -- "
                f"train_loss={entry['train_loss']:.4f}  val_loss={entry['val_loss']:.4f}  val_f1={entry['val_f1']:.3f}"
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["epoch"] for h in seen], y=[h["train_loss"] for h in seen],
                name="train loss", line=dict(color=theme.BLUE, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[h["epoch"] for h in seen], y=[h["val_loss"] for h in seen],
                name="val loss", line=dict(color=theme.RED, width=2),
            ))
            fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
            theme.apply_layout(fig)
            chart_placeholder.plotly_chart(fig, width="stretch")

        state.train(train_config, on_epoch=on_epoch)
        progress_bar.progress(1.0)
        st.success("Training complete.")

    if state.has_artifact():
        st.divider()
        artifact = st.session_state.artifact
        st.subheader("Held-out test metrics (model)")
        metric_row(artifact["test_metrics"], label="Model")
        synthetic_disclaimer()

        st.subheader("Architecture")
        st.markdown(theme.schematic_badge(), unsafe_allow_html=True)
        st.caption("Layer sizes only -- not a live activation view.")
        hidden = st.session_state.train_config.hidden_dim
        layers = [artifact["input_dim"], hidden, max(hidden // 2, 1), 1]
        st.plotly_chart(_schematic_figure(layers), width="stretch")
