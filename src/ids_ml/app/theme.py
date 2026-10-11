"""Dark command-center visual theme: palette, Plotly template, small helpers.

Palette validated with the dataviz skill's validate_palette.js against this
module's own PAGE_BG/SURFACE (not the skill's default charcoal surface) --
categorical set passes lightness/chroma/contrast with CVD in the 8-12 floor
band (legal with direct labels/legend, which the chart helpers below apply).
Status colors pass WCAG contrast against SURFACE.

Every chart built with these helpers must say, in a caption or title,
whether it shows generated ground truth, model inference, or a schematic --
per docs/PRODUCT_VISION.md. See `ground_truth_badge`/`inference_badge`
below for the two recurring cases.
"""

from __future__ import annotations

import plotly.graph_objects as go

PAGE_BG = "#0a0e1a"
SURFACE = "#10152a"
SURFACE_ALT = "#161c35"
BORDER = "rgba(255,255,255,0.08)"

TEXT_PRIMARY = "#f5f7fa"
TEXT_SECONDARY = "#aab4c8"
TEXT_MUTED = "#6b7489"
GRIDLINE = "#22283d"

# Categorical, dark-mode set, slot order fixed (CVD-safety is the ordering).
BLUE = "#3987e5"
TEAL = "#199e70"
AMBER = "#c98500"
GREEN = "#008300"
VIOLET = "#9085e9"
RED = "#e66767"
MAGENTA = "#d55181"
ORANGE = "#d95926"

# Status palette -- fixed roles, never reused for series identity.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
SEVERITY_COLOR = {
    "low": STATUS["good"],
    "medium": STATUS["warning"],
    "high": STATUS["serious"],
    "critical": STATUS["critical"],
}

MODEL_COLOR = BLUE
RULE_COLOR = VIOLET
NORMAL_COLOR = TEAL
SUSPICIOUS_COLOR = RED

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    margin=dict(l=48, r=24, t=48, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=TEXT_SECONDARY),
    ),
    xaxis=dict(
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=BORDER,
        tickfont=dict(color=TEXT_MUTED),
    ),
    yaxis=dict(
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=BORDER,
        tickfont=dict(color=TEXT_MUTED),
    ),
    hoverlabel=dict(bgcolor=SURFACE_ALT, bordercolor=BORDER, font=dict(color=TEXT_PRIMARY)),
)


def apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    layout = {**PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


def inject_css() -> str:
    """Extra CSS the [theme] config.toml can't express (badges, mono labels)."""
    return f"""
    <style>
    .ids-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid {BORDER};
    }}
    .ids-badge-ground-truth {{ color: {TEAL}; border-color: {TEAL}; }}
    .ids-badge-inference {{ color: {BLUE}; border-color: {BLUE}; }}
    .ids-badge-schematic {{ color: {TEXT_MUTED}; border-color: {TEXT_MUTED}; }}
    .ids-badge-rule {{ color: {VIOLET}; border-color: {VIOLET}; }}
    div[data-testid="stMetric"] {{
        background-color: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    </style>
    """


def ground_truth_badge(label: str = "Ground truth") -> str:
    return f'<span class="ids-badge ids-badge-ground-truth">{label}</span>'


def inference_badge(label: str = "Model inference") -> str:
    return f'<span class="ids-badge ids-badge-inference">{label}</span>'


def rule_badge(label: str = "Rule baseline (comparison only)") -> str:
    return f'<span class="ids-badge ids-badge-rule">{label}</span>'


def schematic_badge(label: str = "Schematic") -> str:
    return f'<span class="ids-badge ids-badge-schematic">{label}</span>'
