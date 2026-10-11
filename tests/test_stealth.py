"""Stealth/noisy attack variants: harder to distinguish from benign traffic,
used to test whether a detector generalizes or just memorizes clean
synthetic signatures (tasks/model-maturity item 2).
"""

import numpy as np
import pytest

from ids_ml.data import scenarios
from ids_ml.data.generator import GenerationConfig, generate_dataset
from ids_ml.data.schema import validate

_HOSTS_DF_SEED = 7


def _hosts():
    rng = np.random.default_rng(_HOSTS_DF_SEED)
    return scenarios.generate_hosts(rng, 20)


def test_stealth_mode_requires_boolean():
    with pytest.raises(ValueError, match="stealth_mode"):
        GenerationConfig(seed=1, stealth_mode="yes")


def test_stealth_dataset_is_schema_valid_and_deterministic():
    config = GenerationConfig(
        seed=5, num_hosts=20, duration_minutes=180, background_rate_per_min=60.0,
        port_scan_episodes=2, auth_bruteforce_episodes=2, beaconing_episodes=2,
        service_flood_episodes=2, outbound_transfer_episodes=2, stealth_mode=True,
    )
    df1, _ = generate_dataset(config)
    validate(df1)
    df2, _ = generate_dataset(config)
    import pandas as pd
    pd.testing.assert_frame_equal(df1, df2)


def test_stealth_port_scan_is_slower_and_sparser():
    normal = scenarios.inject_port_scan(np.random.default_rng(1), _hosts(), 7200.0, 0, stealth=False)
    stealth = scenarios.inject_port_scan(np.random.default_rng(1), _hosts(), 7200.0, 0, stealth=True)

    assert len(stealth) < len(normal)
    normal_span = max(r["event_time"] for r in normal) - min(r["event_time"] for r in normal)
    stealth_span = max(r["event_time"] for r in stealth) - min(r["event_time"] for r in stealth)
    assert stealth_span > normal_span


def test_stealth_bruteforce_is_slower_and_sparser():
    normal = scenarios.inject_auth_bruteforce(np.random.default_rng(2), _hosts(), 7200.0, 0, stealth=False)
    stealth = scenarios.inject_auth_bruteforce(np.random.default_rng(2), _hosts(), 7200.0, 0, stealth=True)

    assert len(stealth) < len(normal)
    normal_span = max(r["event_time"] for r in normal) - min(r["event_time"] for r in normal)
    stealth_span = max(r["event_time"] for r in stealth) - min(r["event_time"] for r in stealth)
    assert stealth_span > normal_span


def test_stealth_beaconing_has_larger_jitter():
    normal = scenarios.inject_beaconing(np.random.default_rng(3), _hosts(), 7200.0, 0, stealth=False)
    stealth = scenarios.inject_beaconing(np.random.default_rng(3), _hosts(), 7200.0, 0, stealth=True)

    def gap_std(rows):
        times = sorted(r["event_time"] for r in rows)
        gaps = np.diff(times)
        return float(np.std(gaps)) if len(gaps) > 1 else 0.0

    assert gap_std(stealth) > gap_std(normal)


def test_stealth_service_flood_leans_established():
    stealth = scenarios.inject_service_flood(np.random.default_rng(4), _hosts(), 3600.0, 0, stealth=True)
    normal = scenarios.inject_service_flood(np.random.default_rng(4), _hosts(), 3600.0, 0, stealth=False)

    stealth_established = sum(r["outcome"] == "established" for r in stealth) / len(stealth)
    normal_established = sum(r["outcome"] == "established" for r in normal) / len(normal)
    assert stealth_established > normal_established
    assert len(stealth) < len(normal)
