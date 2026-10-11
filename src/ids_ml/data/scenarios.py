"""Behavioral generators for benign background traffic and attack scenarios.

Each scenario function returns a list of row dicts covering the FEATURE_COLUMNS
plus session_id/src_host_id/dst_host_id/event_time/scenario. flow_id, run_id,
seed, and schema_version are stamped on centrally in generator.py.

These are generated behavioral patterns with documented assumptions, not replays
of real attack traffic. Keep that distinction visible wherever this data is
displayed (see docs/PRODUCT_VISION.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ids_ml.data.schema import HOST_ROLES

HOST_ROLE_WEIGHTS = (0.60, 0.15, 0.25)  # workstation, server, external

_SERVICE_PORTS = {
    "http": 80,
    "https": 443,
    "dns": 53,
    "ssh": 22,
    "rdp": 3389,
    "smtp": 25,
    "ftp": 21,
    "ntp": 123,
}

# (service, protocol, weight) for ordinary background traffic.
_BENIGN_SERVICE_MIX = (
    ("https", "tcp", 0.40),
    ("http", "tcp", 0.15),
    ("dns", "udp", 0.20),
    ("smtp", "tcp", 0.05),
    ("ssh", "tcp", 0.05),
    ("ftp", "tcp", 0.03),
    ("ntp", "udp", 0.05),
    ("other", "tcp", 0.07),
)


def _display_ip(index: int, role: str) -> str:
    """A stable, IP-shaped display identifier -- not used for any logic.

    Internal hosts (workstation/server) get RFC1918 10.0.x.y addresses;
    external hosts get RFC5737 documentation-range addresses (203.0.113.x /
    198.51.100.x) that are guaranteed non-routable, appropriate for a
    synthetic dataset that must never look like it's pointing at a real host.
    """
    if role == "external":
        block = "203.0.113" if index % 2 == 0 else "198.51.100"
        return f"{block}.{(index % 254) + 1}"
    subnet = index // 254
    return f"10.0.{subnet}.{(index % 254) + 1}"


def generate_hosts(rng: np.random.Generator, num_hosts: int) -> pd.DataFrame:
    """Create a coherent host population with role-weighted assignment.

    `host_id` remains the stable internal key everything else groups/joins
    on (see _pick_hosts etc.) -- `ip_address` is purely a display field
    (schema.NON_FEATURE_COLUMNS) for showing analysts something that looks
    like what a real IDS would present.
    """
    if num_hosts < 4:
        raise ValueError("num_hosts must be >= 4 to have enough role diversity")
    roles = rng.choice(HOST_ROLES, size=num_hosts, p=HOST_ROLE_WEIGHTS)
    host_ids = [f"host-{i:04d}" for i in range(num_hosts)]
    ip_addresses = [_display_ip(i, role) for i, role in enumerate(roles)]
    return pd.DataFrame({"host_id": host_ids, "role": roles, "ip_address": ip_addresses})


def _pick_hosts(rng: np.random.Generator, hosts: pd.DataFrame, role: str, n: int = 1) -> list[str]:
    pool = hosts.loc[hosts["role"] == role, "host_id"].to_numpy()
    if len(pool) == 0:
        pool = hosts["host_id"].to_numpy()
    return rng.choice(pool, size=n, replace=True).tolist()


def generate_benign_flows(
    rng: np.random.Generator,
    hosts: pd.DataFrame,
    duration_s: float,
    rate_per_min: float,
) -> list[dict]:
    """Background traffic: internal hosts talking to internal/external peers."""
    internal = hosts.loc[hosts["role"].isin(("workstation", "server")), "host_id"].to_numpy()
    all_hosts = hosts["host_id"].to_numpy()
    n_flows = rng.poisson(rate_per_min * duration_s / 60.0)

    services, protocols, weights = zip(*_BENIGN_SERVICE_MIX)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    rows = []
    for _ in range(n_flows):
        src = rng.choice(internal)
        dst = rng.choice(all_hosts)
        while dst == src:
            dst = rng.choice(all_hosts)
        idx = rng.choice(len(services), p=weights)
        service, protocol = services[idx], protocols[idx]
        dst_port = _SERVICE_PORTS.get(service, int(rng.integers(1024, 65535)))
        duration = float(np.clip(rng.lognormal(mean=0.5, sigma=1.0), 0.01, 600.0))
        packets_sent = max(1, int(rng.lognormal(mean=1.5, sigma=1.0)))
        packets_recv = max(0, int(packets_sent * rng.uniform(0.5, 1.2)))
        bytes_sent = int(packets_sent * rng.uniform(40, 1200))
        bytes_recv = int(packets_recv * rng.uniform(40, 1400))
        outcome = rng.choice(
            ("established", "reset", "timeout"), p=(0.92, 0.05, 0.03)
        )
        rows.append(
            {
                "event_time": float(rng.uniform(0, duration_s)),
                "session_id": str(src),
                "src_host_id": str(src),
                "dst_host_id": str(dst),
                "src_port": int(rng.integers(1024, 65535)),
                "dst_port": int(dst_port),
                "protocol": protocol,
                "service": service,
                "duration_s": duration,
                "packets_sent": packets_sent,
                "packets_recv": packets_recv,
                "bytes_sent": bytes_sent,
                "bytes_recv": bytes_recv,
                "outcome": str(outcome),
                "scenario": "benign",
            }
        )
    return rows


def _distinct_pair(
    rng: np.random.Generator, hosts: pd.DataFrame, source_role: str, dest_role: str
) -> tuple[str, str]:
    """Prefer role-consistent peers, with distinct endpoints even in tiny pools."""
    source = _pick_hosts(rng, hosts, source_role)[0]
    pool = hosts.loc[(hosts.role == dest_role) & (hosts.host_id != source), "host_id"]
    if pool.empty:
        pool = hosts.loc[hosts.host_id != source, "host_id"]
    return str(source), str(rng.choice(pool.to_numpy()))


def _wire_bytes(rng: np.random.Generator, packets: int, mean_size: float) -> int:
    """Synthetic aggregate IP bytes: variable mean packet size, bounded by MTU.

    Assumes unfragmented <=1500-byte packets with a 40-byte accounting floor.
    This is a simulation convention, not a fit to measured traffic.
    """
    if not packets:
        return 0
    size = float(np.clip(rng.lognormal(np.log(mean_size), 0.25), 40, 1500))
    return int(packets * size)


def refine_background_flows(
    rng: np.random.Generator, rows: list[dict], duration_s: float
) -> None:
    """Opt-in service-aware counters and durations; preserve row identities/times.

    DNS/NTP are short request/response exchanges; web/transfer services have
    heavier tails and asymmetric payloads. No ground-truth attack is modified.
    """
    for row in rows:
        if row["scenario"] != "benign":
            continue
        service = row["service"]
        if service in ("dns", "ntp"):
            sent = int(rng.integers(1, 4))
            received = sent
            sent_size, received_size = (80, 220) if service == "dns" else (76, 76)
            duration = float(rng.lognormal(-3, 0.7))
        else:
            sent = max(3, int(rng.lognormal(2.2, 1.1)))
            received = max(2, int(sent * rng.lognormal(0.3, 0.5)))
            sent_size, received_size = {
                "http": (180, 1000), "https": (250, 1100),
                "ftp": (1100, 180), "smtp": (750, 150),
                "ssh": (160, 200),
            }.get(service, (350, 500))
            duration = float(rng.lognormal(0.4, 1.0)) + (sent + received) / float(rng.uniform(20, 300))
        if row["outcome"] == "timeout":
            received = 0
            duration = float(rng.uniform(1, 10))
        elif row["outcome"] == "reset":
            sent, received = min(sent, 3), min(received, 2)
            sent_size = received_size = 60
        row.update(
            packets_sent=sent, packets_recv=received,
            bytes_sent=_wire_bytes(rng, sent, sent_size),
            bytes_recv=_wire_bytes(rng, received, received_size),
            duration_s=max(0.0, min(duration, duration_s - row["event_time"])),
        )


def inject_service_flood(
    rng: np.random.Generator, hosts: pd.DataFrame, duration_s: float, episode_idx: int,
    *, stealth: bool = False,
) -> list[dict]:
    """Bursty connections from an external peer to one HTTP(S) server.

    A mixture of incomplete and established connections models congestion;
    this is flow telemetry, not emitted packets or an actual denial of service.

    stealth=True: lower volume, spread over minutes instead of seconds, and
    mostly-established outcomes -- tests whether a detector is keying on
    raw burst volume/incompleteness rather than the underlying pattern.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    source, dest = _distinct_pair(rng, hosts, "external", "server")
    service = str(rng.choice(("http", "https")))
    count = int(rng.integers(20, 60)) if stealth else int(rng.integers(100, 250))
    span = min(duration_s, float(rng.uniform(30, 120)) if stealth else float(rng.uniform(2, 15)))
    start = float(rng.uniform(0, max(0, duration_s - span)))
    gaps = rng.lognormal(-2, 1.0, count)
    times = start + span * np.cumsum(gaps) / (gaps.sum() + gaps[-1])
    outcome_p = (0.7, 0.2, 0.1) if stealth else (0.3, 0.5, 0.2)
    rows = []
    for time in times:
        outcome = str(rng.choice(("established", "timeout", "reset"), p=outcome_p))
        sent = int(rng.integers(2, 9))
        received = 0 if outcome == "timeout" else int(rng.integers(1, sent + 1))
        duration = float(rng.lognormal(-2, 0.7)) if outcome != "timeout" else float(rng.uniform(0.5, 3))
        rows.append(dict(
            event_time=float(time), session_id=f"service_flood-{episode_idx:05d}",
            src_host_id=source, dst_host_id=dest,
            src_port=int(rng.integers(1024, 65535)), dst_port=_SERVICE_PORTS[service],
            protocol="tcp", service=service,
            duration_s=min(duration, duration_s - float(time)),
            packets_sent=sent, packets_recv=received,
            bytes_sent=_wire_bytes(rng, sent, 80), bytes_recv=_wire_bytes(rng, received, 100),
            outcome=outcome, scenario="service_flood",
        ))
    return rows


def inject_outbound_transfer(
    rng: np.random.Generator, hosts: pd.DataFrame, duration_s: float, episode_idx: int,
    *, stealth: bool = False,
) -> list[dict]:
    """Chunked, upload-heavy HTTPS transfers from one host to an external peer.

    Similar benign uploads can have the same flow signature: provenance is
    ground truth for this generated episode, not proof of malicious intent.

    stealth=True: lower throughput spread over a much longer window --
    exfiltration paced to blend into ordinary background upload traffic.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    source, dest = _distinct_pair(rng, hosts, "workstation", "external")
    count = int(rng.integers(8, 25))
    span_range = (900, 3600) if stealth else (60, 600)
    span = min(duration_s, float(rng.uniform(*span_range)))
    start = float(rng.uniform(0, max(0, duration_s - span)))
    # Consecutive chunk windows share a sampled throughput and jittered sizes.
    weights = rng.lognormal(0, 0.45, count)
    windows = span * weights / weights.sum()
    throughput_mean = 120 if stealth else 500
    throughput = float(rng.lognormal(np.log(throughput_mean), 0.5))  # packets per second
    rows = []
    time = start
    for window in windows:
        duration = float(window * rng.uniform(0.65, 0.95))
        sent = max(10, int(duration * throughput))
        received = max(2, int(sent * rng.uniform(0.3, 0.6)))
        rows.append(dict(
            event_time=float(time), session_id=f"outbound_transfer-{episode_idx:05d}",
            src_host_id=source, dst_host_id=dest,
            src_port=int(rng.integers(1024, 65535)), dst_port=443,
            protocol="tcp", service="https", duration_s=duration,
            packets_sent=sent, packets_recv=received,
            bytes_sent=_wire_bytes(rng, sent, 1200), bytes_recv=_wire_bytes(rng, received, 60),
            outcome="established", scenario="outbound_transfer",
        ))
        time += float(window)
    return rows


def inject_port_scan(
    rng: np.random.Generator,
    hosts: pd.DataFrame,
    duration_s: float,
    episode_idx: int,
    *, stealth: bool = False,
) -> list[dict]:
    """One host sweeping many ports on a victim in a short, noisy burst.

    stealth=True: far fewer ports, spread over tens of minutes instead of
    under a minute -- a "low and slow" scan meant to stay under
    rate-based/volume thresholds.
    """
    attacker = _pick_hosts(rng, hosts, "workstation", 1)[0]
    victim_pool = hosts.loc[hosts["role"] == "server", "host_id"].to_numpy()
    victim = str(rng.choice(victim_pool)) if len(victim_pool) else _pick_hosts(rng, hosts, "server", 1)[0]

    n_ports = int(rng.integers(10, 40)) if stealth else int(rng.integers(30, 200))
    ports = rng.choice(np.arange(1, 49152), size=n_ports, replace=False)
    burst_span = float(rng.uniform(300, 1800)) if stealth else float(rng.uniform(5, 60))
    burst_start = float(rng.uniform(0, max(duration_s - burst_span, 1)))

    rows = []
    for port in ports:
        t = burst_start + float(rng.uniform(0, burst_span))
        outcome = rng.choice(("rejected", "timeout"), p=(0.7, 0.3))
        rows.append(
            {
                "event_time": t,
                "session_id": f"port_scan-{episode_idx:05d}",
                "src_host_id": attacker,
                "dst_host_id": victim,
                "src_port": int(rng.integers(1024, 65535)),
                "dst_port": int(port),
                "protocol": "tcp",
                "service": "other",
                "duration_s": float(rng.uniform(0.01, 0.5)),
                "packets_sent": int(rng.integers(1, 3)),
                "packets_recv": int(rng.integers(0, 2)),
                "bytes_sent": int(rng.integers(40, 120)),
                "bytes_recv": int(rng.integers(0, 80)),
                "outcome": str(outcome),
                "scenario": "port_scan",
            }
        )
    return rows


def inject_auth_bruteforce(
    rng: np.random.Generator,
    hosts: pd.DataFrame,
    duration_s: float,
    episode_idx: int,
    *, stealth: bool = False,
) -> list[dict]:
    """Many failed auth attempts from one source against one service port.

    stealth=True: far fewer attempts spread over hours -- "low and slow"
    credential stuffing meant to stay under per-minute attempt thresholds.
    """
    attacker = _pick_hosts(rng, hosts, "external", 1)[0]
    victim = _pick_hosts(rng, hosts, "server", 1)[0]
    service = str(rng.choice(("ssh", "rdp")))
    port = _SERVICE_PORTS[service]

    n_attempts = int(rng.integers(5, 20)) if stealth else int(rng.integers(30, 150))
    burst_span = float(rng.uniform(1800, 7200)) if stealth else float(rng.uniform(30, 300))
    burst_start = float(rng.uniform(0, max(duration_s - burst_span, 1)))

    rows = []
    for i in range(n_attempts):
        t = burst_start + (burst_span * i / n_attempts) + float(rng.uniform(0, 1.0))
        outcome = rng.choice(("rejected", "reset"), p=(0.85, 0.15))
        rows.append(
            {
                "event_time": t,
                "session_id": f"auth_bruteforce-{episode_idx:05d}",
                "src_host_id": attacker,
                "dst_host_id": victim,
                "src_port": int(rng.integers(1024, 65535)),
                "dst_port": int(port),
                "protocol": "tcp",
                "service": service,
                "duration_s": float(rng.uniform(0.05, 1.5)),
                "packets_sent": int(rng.integers(2, 6)),
                "packets_recv": int(rng.integers(1, 4)),
                "bytes_sent": int(rng.integers(80, 300)),
                "bytes_recv": int(rng.integers(40, 200)),
                "outcome": str(outcome),
                "scenario": "auth_bruteforce",
            }
        )
    return rows


def inject_beaconing(
    rng: np.random.Generator,
    hosts: pd.DataFrame,
    duration_s: float,
    episode_idx: int,
    *, stealth: bool = False,
) -> list[dict]:
    """A compromised-looking internal host checking in with a fixed external peer.

    stealth=True: a longer, more irregular interval (jitter fraction ~3.5x
    larger) -- C2 designed to avoid a fixed-period fingerprint, which is
    exactly the regularity a naive detector (rule-based or otherwise) would
    key on.
    """
    source = _pick_hosts(rng, hosts, "workstation", 1)[0]
    dest = _pick_hosts(rng, hosts, "external", 1)[0]
    interval = float(rng.uniform(180, 600)) if stealth else float(rng.uniform(45, 120))
    jitter_frac = 0.35 if stealth else 0.1
    byte_size = int(rng.integers(200, 600))
    n_beacons = max(3, int(duration_s / interval))

    rows = []
    t = float(rng.uniform(0, interval))
    for _ in range(n_beacons):
        if t > duration_s:
            break
        rows.append(
            {
                "event_time": t,
                "session_id": f"beaconing-{episode_idx:05d}",
                "src_host_id": source,
                "dst_host_id": dest,
                "src_port": int(rng.integers(1024, 65535)),
                "dst_port": 443,
                "protocol": "tcp",
                "service": "https",
                "duration_s": float(rng.uniform(0.2, 1.0)),
                "packets_sent": int(rng.integers(3, 6)),
                "packets_recv": int(rng.integers(3, 6)),
                "bytes_sent": byte_size + int(rng.integers(-20, 20)),
                "bytes_recv": byte_size + int(rng.integers(-20, 20)),
                "outcome": "established",
                "scenario": "beaconing",
            }
        )
        t += interval + float(rng.normal(0, interval * jitter_frac))
    return rows
