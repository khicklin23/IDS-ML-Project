"""Seeded synthetic IDS dataset generator.

Produces a reproducible mix of benign background traffic and a small set of
well-tested attack scenario families (see scenarios.py), stamped with a
generation manifest for provenance. This is generated behavioral data with
documented assumptions, not a measurement of real network traffic or
real-world intrusion detection accuracy.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ids_ml.data import scenarios
from ids_ml.data.schema import SCENARIO_LABEL, SCHEMA_VERSION, coerce_dtypes, validate


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    seed: int
    num_hosts: int = 40
    duration_minutes: float = 60.0
    background_rate_per_min: float = 60.0
    port_scan_episodes: int = 3
    auth_bruteforce_episodes: int = 3
    beaconing_episodes: int = 3
    service_flood_episodes: int = 0
    outbound_transfer_episodes: int = 0
    realistic_background: bool = False
    stealth_mode: bool = False

    def __post_init__(self) -> None:
        for field in ("service_flood_episodes", "outbound_transfer_episodes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not isinstance(self.realistic_background, bool):
            raise ValueError("realistic_background must be a bool")
        if not isinstance(self.stealth_mode, bool):
            raise ValueError("stealth_mode must be a bool")

    @property
    def duration_s(self) -> float:
        return self.duration_minutes * 60.0

    @property
    def run_id(self) -> str:
        return f"run-{self.seed}"


def generate_dataset(config: GenerationConfig) -> tuple[pd.DataFrame, dict]:
    """Generate a schema-validated flow DataFrame and its manifest.

    Deterministic: the same config always produces an identical DataFrame
    (see tests/test_generator.py::test_determinism).
    """
    rng = np.random.default_rng(config.seed)
    hosts = scenarios.generate_hosts(rng, config.num_hosts)

    rows: list[dict] = []
    rows.extend(
        scenarios.generate_benign_flows(
            rng, hosts, config.duration_s, config.background_rate_per_min
        )
    )
    stealth = config.stealth_mode
    for i in range(config.port_scan_episodes):
        rows.extend(scenarios.inject_port_scan(rng, hosts, config.duration_s, i, stealth=stealth))
    for i in range(config.auth_bruteforce_episodes):
        rows.extend(scenarios.inject_auth_bruteforce(rng, hosts, config.duration_s, i, stealth=stealth))
    for i in range(config.beaconing_episodes):
        rows.extend(scenarios.inject_beaconing(rng, hosts, config.duration_s, i, stealth=stealth))

    # Opt-ins consume no additional random draws when disabled, preserving
    # existing datasets and the UI's default demo exactly.
    if config.realistic_background:
        scenarios.refine_background_flows(rng, rows, config.duration_s)
    for i in range(config.service_flood_episodes):
        rows.extend(scenarios.inject_service_flood(rng, hosts, config.duration_s, i, stealth=stealth))
    for i in range(config.outbound_transfer_episodes):
        rows.extend(scenarios.inject_outbound_transfer(rng, hosts, config.duration_s, i, stealth=stealth))

    if not rows:
        raise ValueError("generation produced zero rows; check config rates/episode counts")

    df = pd.DataFrame(rows)
    df = df.sort_values("event_time", kind="stable").reset_index(drop=True)

    df["schema_version"] = SCHEMA_VERSION
    df["run_id"] = config.run_id
    df["seed"] = config.seed
    df["flow_id"] = [f"{config.run_id}-{i:07d}" for i in range(len(df))]
    df["label"] = df["scenario"].map(SCENARIO_LABEL)

    ip_by_host = hosts.set_index("host_id")["ip_address"]
    df["src_ip"] = df["src_host_id"].map(ip_by_host)
    df["dst_ip"] = df["dst_host_id"].map(ip_by_host)

    # Inter-arrival time within each session -- real IDS telemetry would
    # compute this the same way (consecutive flows sharing a source/session).
    df["prior_gap_s"] = df.groupby("session_id")["event_time"].diff().fillna(0.0)

    df = coerce_dtypes(df)
    validate(df)

    manifest = build_manifest(config, df)
    return df, manifest


def build_manifest(config: GenerationConfig, df: pd.DataFrame) -> dict:
    scenario_counts = df["scenario"].value_counts().to_dict()
    label_counts = df["label"].value_counts().to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": dataclasses.asdict(config),
        "row_count": int(len(df)),
        "host_count": int(config.num_hosts),
        "scenario_counts": {str(k): int(v) for k, v in scenario_counts.items()},
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }


def save_dataset(df: pd.DataFrame, manifest: dict, out_dir: Path | str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "flows.parquet"
    manifest_path = out_dir / "manifest.json"
    df.to_parquet(data_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"data": data_path, "manifest": manifest_path}


def load_dataset(out_dir: Path | str) -> tuple[pd.DataFrame, dict]:
    out_dir = Path(out_dir)
    df = pd.read_parquet(out_dir / "flows.parquet")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"dataset schema_version {manifest['schema_version']!r} does not match "
            f"current SCHEMA_VERSION {SCHEMA_VERSION!r}"
        )
    return df, manifest


def split_dataset(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-wise split by session_id so no episode/host-session leaks across splits.

    Benign background rows are grouped by src_host_id (their session_id);
    each attack episode is its own session_id. Splitting whole groups keeps
    related/duplicated flows out of more than one split, as the schema and
    vision require.

    Allocation is done per scenario family (every session is 100% one
    scenario by construction), not by shuffling all sessions together.
    Without this, a scenario with few episodes relative to the number of
    benign host-sessions could land zero val/test rows purely by chance --
    this bit the Streamlit UI default, a multiclass test fixture, and a
    showcase run before being fixed here. A scenario with only 1-2 sessions
    still can't be fully represented in all three splits (there's nothing
    to allocate); those get a best-effort split documented in-line below
    rather than crashing.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    rng = np.random.default_rng(seed)
    session_scenario = df.groupby("session_id")["scenario"].first().astype(str)

    train_sessions: set = set()
    val_sessions: set = set()
    test_sessions: set = set()

    for scenario_name in session_scenario.unique():
        sessions = session_scenario.index[session_scenario == scenario_name].to_numpy(dtype=object)
        rng.shuffle(sessions)
        n = len(sessions)

        n_test = min(max(1, round(n * test_frac)), n)
        n_val = min(max(1, round(n * val_frac)), n - n_test) if n > n_test else 0

        test_sessions.update(sessions[:n_test])
        val_sessions.update(sessions[n_test : n_test + n_val])
        train_sessions.update(sessions[n_test + n_val :])

    train_df = df[df["session_id"].isin(train_sessions)].reset_index(drop=True)
    val_df = df[df["session_id"].isin(val_sessions)].reset_index(drop=True)
    test_df = df[df["session_id"].isin(test_sessions)].reset_index(drop=True)

    assert not (train_sessions & val_sessions & test_sessions)
    return train_df, val_df, test_df
