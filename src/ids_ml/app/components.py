"""Small shared UI helpers used by more than one page."""

from __future__ import annotations

import streamlit as st

from ids_ml.app import state, theme


def require_dataset() -> bool:
    if not state.has_dataset():
        st.info("No dataset yet. Go to **Scenario & Data Lab** and generate one first.")
        return False
    return True


def require_artifact() -> bool:
    if not require_dataset():
        return False
    if not state.has_artifact():
        st.info("No trained model yet. Go to **Training Studio** and train one first.")
        return False
    return True


def require_scores() -> bool:
    if not require_artifact():
        return False
    if not state.has_scores():
        st.info("Model trained but not scored yet. Go to **Detection & Replay** and score the held-out set.")
        return False
    return True


def metric_row(metrics: dict, *, label: str) -> None:
    cols = st.columns(5)
    fields = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("False positive rate", "false_positive_rate"),
    ]
    for col, (title, key) in zip(cols, fields):
        col.metric(f"{label} — {title}", f"{metrics[key]:.3f}")


def synthetic_disclaimer() -> None:
    st.caption(
        "Synthetic evaluation on generated data only. Does not establish "
        "real-world intrusion detection accuracy."
    )
