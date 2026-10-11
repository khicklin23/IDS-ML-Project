"""Versioned schema for synthetic IDS flow/event records.

Bumping SCHEMA_VERSION is required whenever a column is added, removed,
renamed, or changes meaning. Generation manifests record the version that
produced them so downstream code can refuse mismatched data instead of
silently misinterpreting columns.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = "1.2.0"  # 1.2.0 adds prior_gap_s feature (additive)

PROTOCOLS = ("tcp", "udp", "icmp")
SERVICES = ("http", "https", "dns", "ssh", "rdp", "smtp", "ftp", "ntp", "other")
OUTCOMES = ("established", "rejected", "timeout", "reset")
HOST_ROLES = ("workstation", "server", "external")
SCENARIOS = ("benign", "port_scan", "auth_bruteforce", "beaconing", "service_flood", "outbound_transfer")
LABELS = ("normal", "suspicious")

SCENARIO_LABEL = {
    "benign": "normal",
    "port_scan": "suspicious",
    "auth_bruteforce": "suspicious",
    "beaconing": "suspicious",
    "service_flood": "suspicious",
    "outbound_transfer": "suspicious",
}

# Columns a model may legitimately learn from. Host identity, run/seed
# provenance, scenario labels, and grouping keys are deliberately excluded
# below (see NON_FEATURE_COLUMNS) so the generator's own bookkeeping can
# never leak into the detector as a shortcut feature.
FEATURE_COLUMNS = (
    "duration_s",
    "src_port",
    "dst_port",
    "protocol",
    "service",
    "packets_sent",
    "packets_recv",
    "bytes_sent",
    "bytes_recv",
    "outcome",
    # Seconds since the previous flow in the same session_id (0.0 for the
    # first flow in a session). A standard, real-world-derivable NetFlow
    # feature (inter-arrival time), not generator-only bookkeeping -- it's
    # computed from event_time + session_id, same as a real collector could.
    # Added because it's beaconing's actual discriminating signal (regular
    # intervals) that no single-flow feature otherwise captures -- see
    # tasks/model-maturity for the evaluation that motivated this.
    "prior_gap_s",
)

TARGET_COLUMN = "label"

# Generator-only bookkeeping, ground-truth provenance, and split grouping
# keys. Never pass these into a model's input features. src_ip/dst_ip are
# purely a display convenience (an IDS analyst expects to see addresses,
# not opaque host ids) -- same leakage risk as host identity, so they stay
# out of FEATURE_COLUMNS for the same reason.
NON_FEATURE_COLUMNS = (
    "schema_version",
    "run_id",
    "seed",
    "flow_id",
    "event_time",
    "src_host_id",
    "dst_host_id",
    "src_ip",
    "dst_ip",
    "session_id",
    "scenario",
    TARGET_COLUMN,
)

ALL_COLUMNS = (
    "schema_version",
    "run_id",
    "seed",
    "flow_id",
    "event_time",
    "session_id",
    "src_host_id",
    "dst_host_id",
    "src_ip",
    "dst_ip",
) + FEATURE_COLUMNS + (
    "scenario",
    TARGET_COLUMN,
)

_DTYPES: dict[str, str] = {
    "schema_version": "string",
    "run_id": "string",
    "seed": "int64",
    "flow_id": "string",
    "event_time": "float64",
    "session_id": "string",
    "src_host_id": "string",
    "dst_host_id": "string",
    "src_ip": "string",
    "dst_ip": "string",
    "duration_s": "float64",
    "src_port": "int64",
    "dst_port": "int64",
    "protocol": "category",
    "service": "category",
    "packets_sent": "int64",
    "packets_recv": "int64",
    "bytes_sent": "int64",
    "bytes_recv": "int64",
    "outcome": "category",
    "prior_gap_s": "float64",
    "scenario": "category",
    "label": "category",
}


class SchemaError(ValueError):
    """Raised when a DataFrame violates the versioned flow schema."""


def validate(df: pd.DataFrame, *, schema_version: str = SCHEMA_VERSION) -> None:
    """Validate structural and value constraints of a flow DataFrame.

    Raises SchemaError on any violation; callers decide whether to catch,
    log, or let it propagate. Intended for generator self-checks and tests,
    not as runtime UI validation.
    """
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    if df.empty:
        raise SchemaError("empty DataFrame")

    if (df["schema_version"] != schema_version).any():
        bad = sorted(df.loc[df["schema_version"] != schema_version, "schema_version"].unique())
        raise SchemaError(f"schema_version mismatch, expected {schema_version!r}, found {bad}")

    if df[list(ALL_COLUMNS)].isna().any().any():
        cols = df[list(ALL_COLUMNS)].columns[df[list(ALL_COLUMNS)].isna().any()].tolist()
        raise SchemaError(f"null values in required columns: {cols}")

    if df["flow_id"].duplicated().any():
        raise SchemaError("duplicate flow_id values")

    _check_enum(df, "protocol", PROTOCOLS)
    _check_enum(df, "service", SERVICES)
    _check_enum(df, "outcome", OUTCOMES)
    _check_enum(df, "scenario", SCENARIOS)
    _check_enum(df, "label", LABELS)

    for col in ("duration_s", "packets_sent", "packets_recv", "bytes_sent", "bytes_recv"):
        if (df[col] < 0).any():
            raise SchemaError(f"negative values in {col}")

    for col in ("src_port", "dst_port"):
        if ((df[col] < 0) | (df[col] > 65535)).any():
            raise SchemaError(f"{col} out of valid port range")

    mismatched = df["label"].astype(str) != df["scenario"].astype(str).map(SCENARIO_LABEL)
    if mismatched.any():
        raise SchemaError(f"label does not match scenario->label mapping on {mismatched.sum()} rows")

    if not (df["event_time"].sort_values().reset_index(drop=True) == df["event_time"]).all():
        pass  # ordering is a convenience, not a correctness requirement; no-op kept explicit.


def _check_enum(df: pd.DataFrame, column: str, allowed: tuple[str, ...]) -> None:
    values = set(df[column].astype(str).unique())
    extra = values - set(allowed)
    if extra:
        raise SchemaError(f"{column} has values outside {allowed}: {sorted(extra)}")


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the schema's canonical dtypes to a freshly built DataFrame."""
    return df.astype({k: v for k, v in _DTYPES.items() if k in df.columns})
