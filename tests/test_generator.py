import pandas as pd
import pytest

from ids_ml.data.generator import GenerationConfig, generate_dataset, split_dataset
from ids_ml.data.schema import FEATURE_COLUMNS, SCENARIOS, validate

SMALL_CONFIG = GenerationConfig(
    seed=42,
    num_hosts=12,
    duration_minutes=10,
    background_rate_per_min=150.0,
    port_scan_episodes=2,
    auth_bruteforce_episodes=2,
    beaconing_episodes=2,
)


def test_generated_dataset_is_schema_valid():
    df, manifest = generate_dataset(SMALL_CONFIG)
    validate(df)  # should not raise
    assert manifest["row_count"] == len(df)
    assert set(manifest["scenario_counts"]) <= set(SCENARIOS)


def test_determinism_same_seed_same_output():
    df1, _ = generate_dataset(SMALL_CONFIG)
    df2, _ = generate_dataset(SMALL_CONFIG)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_differs():
    df1, _ = generate_dataset(SMALL_CONFIG)
    other = GenerationConfig(**{**vars(SMALL_CONFIG), "seed": 43})
    df2, _ = generate_dataset(other)
    assert not df1["bytes_sent"].equals(df2["bytes_sent"])


def test_all_scenario_families_present():
    df, _ = generate_dataset(SMALL_CONFIG)
    present = set(df["scenario"].astype(str).unique())
    assert present == {"benign", "port_scan", "auth_bruteforce", "beaconing"}


def test_suspicious_rows_exist_and_are_minority():
    df, _ = generate_dataset(SMALL_CONFIG)
    label_counts = df["label"].value_counts()
    assert label_counts.get("suspicious", 0) > 0
    assert label_counts.get("normal", 0) > label_counts.get("suspicious", 0)


def test_zero_rows_raises():
    empty_config = GenerationConfig(
        seed=1,
        num_hosts=4,
        duration_minutes=1,
        background_rate_per_min=0.0,
        port_scan_episodes=0,
        auth_bruteforce_episodes=0,
        beaconing_episodes=0,
    )
    with pytest.raises(ValueError):
        generate_dataset(empty_config)


def test_split_has_no_session_overlap():
    df, _ = generate_dataset(SMALL_CONFIG)
    train, val, test = split_dataset(df, val_frac=0.2, test_frac=0.2, seed=7)

    train_sessions = set(train["session_id"])
    val_sessions = set(val["session_id"])
    test_sessions = set(test["session_id"])

    assert not (train_sessions & val_sessions)
    assert not (train_sessions & test_sessions)
    assert not (val_sessions & test_sessions)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_is_deterministic():
    df, _ = generate_dataset(SMALL_CONFIG)
    a = split_dataset(df, val_frac=0.2, test_frac=0.2, seed=7)
    b = split_dataset(df, val_frac=0.2, test_frac=0.2, seed=7)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right)


def test_feature_columns_exclude_identity_and_provenance():
    leaky = {
        "src_host_id", "dst_host_id", "src_ip", "dst_ip",
        "scenario", "label", "run_id", "seed", "session_id",
    }
    assert leaky.isdisjoint(FEATURE_COLUMNS)


def test_split_is_stratified_by_scenario_across_many_seeds():
    """Regression test: split_dataset used to shuffle all sessions together
    with no label stratification, so a scenario with few episodes relative
    to the benign host-session count could land zero val/test rows purely
    by chance. This hit the Streamlit UI default, a multiclass test fixture,
    and a showcase run before being fixed -- every scenario present in the
    full dataset must now get at least one session in every split, for
    every seed, not just a hand-picked lucky one.
    """
    config = GenerationConfig(
        seed=0, num_hosts=30, duration_minutes=60, background_rate_per_min=60.0,
        port_scan_episodes=4, auth_bruteforce_episodes=4, beaconing_episodes=4,
        service_flood_episodes=4, outbound_transfer_episodes=4,
    )
    df, _ = generate_dataset(config)
    present_scenarios = set(df["scenario"].astype(str).unique())
    assert len(present_scenarios) > 1  # sanity: this config actually has multiple families

    for seed in range(20):
        train, val, test = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=seed)
        for name, split_df in (("train", train), ("val", val), ("test", test)):
            missing = present_scenarios - set(split_df["scenario"].astype(str).unique())
            assert not missing, f"seed={seed} {name} split missing scenarios: {missing}"


def _expanded_config(**overrides):
    return GenerationConfig(**{
        **vars(SMALL_CONFIG), "service_flood_episodes": 2,
        "outbound_transfer_episodes": 2, "realistic_background": True,
        **overrides,
    })


def test_expanded_registry_determinism_and_manifest():
    config = _expanded_config()
    first, manifest = generate_dataset(config)
    second, _ = generate_dataset(config)
    pd.testing.assert_frame_equal(first, second)
    validate(first)
    assert set(first.scenario.astype(str)) == set(SCENARIOS)
    assert manifest["config"]["realistic_background"] is True
    for family in ("service_flood", "outbound_transfer"):
        subset = first[first.scenario == family]
        assert len(subset) == manifest["scenario_counts"][family]
        assert subset.session_id.nunique() == 2
        assert (subset.label == "suspicious").all()


def test_service_flood_is_bursty_with_consistent_endpoints_and_outcomes():
    df, _ = generate_dataset(_expanded_config())
    for _, episode in df[df.scenario == "service_flood"].groupby("session_id"):
        assert episode.src_host_id.nunique() == episode.dst_host_id.nunique() == 1
        assert (episode.src_host_id != episode.dst_host_id).all()
        assert episode.event_time.max() - episode.event_time.min() <= 15
        assert set(episode.outcome.astype(str)) == {"established", "timeout", "reset"}
        assert (episode.loc[episode.outcome == "timeout", "bytes_recv"] == 0).all()
        assert (episode.service.map({"http": 80, "https": 443}).astype(int) == episode.dst_port).all()
        gaps = episode.event_time.diff().dropna()
        assert gaps.std() > 0  # variable inter-arrival time, not a fixed cadence


def test_outbound_transfer_has_upload_asymmetry_and_sequential_chunks():
    df, _ = generate_dataset(_expanded_config())
    for _, episode in df[df.scenario == "outbound_transfer"].groupby("session_id"):
        assert episode.src_host_id.nunique() == episode.dst_host_id.nunique() == 1
        assert (episode.src_host_id != episode.dst_host_id).all()
        assert (episode.bytes_sent > 3 * episode.bytes_recv).all()
        assert (episode.outcome == "established").all()
        assert (episode.dst_port == 443).all()
        ends = episode.event_time + episode.duration_s
        assert (ends.iloc[:-1].to_numpy() <= episode.event_time.iloc[1:].to_numpy()).all()


def test_realistic_counters_and_service_distributions():
    config = _expanded_config(background_rate_per_min=500)
    df, _ = generate_dataset(config)
    refined = df[df.scenario.isin(["benign", "service_flood", "outbound_transfer"])]
    for direction in ("sent", "recv"):
        packets, byte_counts = refined[f"packets_{direction}"], refined[f"bytes_{direction}"]
        assert (byte_counts >= 40 * packets).all()
        assert (byte_counts <= 1500 * packets).all()
        assert (byte_counts[packets == 0] == 0).all()
    assert (refined.event_time >= 0).all()
    assert (refined.event_time + refined.duration_s <= config.duration_s + 1e-9).all()
    normal = df[(df.scenario == "benign") & (df.outcome == "established")]
    dns, web = normal[normal.service == "dns"], normal[normal.service == "https"]
    assert dns.packets_sent.max() <= 3
    assert dns.duration_s.median() < web.duration_s.median()
    assert (web.bytes_recv / web.packets_recv).median() > (dns.bytes_recv / dns.packets_recv).median()


def test_expanded_sessions_never_straddle_splits():
    df, _ = generate_dataset(_expanded_config())
    splits = split_dataset(df, seed=7)
    for family in ("service_flood", "outbound_transfer"):
        for session in df.loc[df.scenario == family, "session_id"].unique():
            assert sum(session in set(part.session_id) for part in splits) == 1
    assert sum(map(len, splits)) == len(df)


def test_disabled_options_preserve_legacy_random_stream():
    import numpy as np
    from ids_ml.data import scenarios

    rng = np.random.default_rng(SMALL_CONFIG.seed)
    hosts = scenarios.generate_hosts(rng, SMALL_CONFIG.num_hosts)
    rows = scenarios.generate_benign_flows(
        rng, hosts, SMALL_CONFIG.duration_s, SMALL_CONFIG.background_rate_per_min,
    )
    for function, count in (
        (scenarios.inject_port_scan, SMALL_CONFIG.port_scan_episodes),
        (scenarios.inject_auth_bruteforce, SMALL_CONFIG.auth_bruteforce_episodes),
        (scenarios.inject_beaconing, SMALL_CONFIG.beaconing_episodes),
    ):
        for episode in range(count):
            rows.extend(function(rng, hosts, SMALL_CONFIG.duration_s, episode))
    expected = pd.DataFrame(rows).sort_values("event_time", kind="stable").reset_index(drop=True)
    actual, _ = generate_dataset(SMALL_CONFIG)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, check_categorical=False)


@pytest.mark.parametrize("function_name", ["inject_service_flood", "inject_outbound_transfer"])
@pytest.mark.parametrize("duration", [0.05, 1.0, 60.0])
def test_new_episodes_fit_short_windows_and_missing_role_pools(function_name, duration):
    import numpy as np
    from ids_ml.data import scenarios

    hosts = pd.DataFrame({"host_id": ["a", "b", "c", "d"], "role": ["workstation"] * 4})
    function = getattr(scenarios, function_name)
    rows = function(np.random.default_rng(4), hosts, duration, 0)
    assert rows
    assert all(row["src_host_id"] != row["dst_host_id"] for row in rows)
    assert all(0 <= row["event_time"] < duration for row in rows)
    assert all(row["event_time"] + row["duration_s"] <= duration + 1e-9 for row in rows)


@pytest.mark.parametrize("field", ["service_flood_episodes", "outbound_transfer_episodes"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_new_episode_counts_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        GenerationConfig(seed=4, **{field: value})


def test_background_flag_requires_boolean():
    with pytest.raises(ValueError, match="realistic_background"):
        GenerationConfig(seed=4, realistic_background="false")
