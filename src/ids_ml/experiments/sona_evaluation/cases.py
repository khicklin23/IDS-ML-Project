"""Independent case generator for the Sona evaluation suite.

Written separately from the candidate's and earlier studies' generators. Each
window is one host's ordered flows. Generator labels live in a key table and
in training-only columns; `inference_frame` removes them before any adapter
sees the data. Identifiers are random hex, so they encode nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ids_ml.data.schema import SCHEMA_VERSION, validate
from .protocol import (ATTACK_KINDS, CONDITIONS, FAMILIAR_GAPS, INFERENCE_COLUMNS,
                       LOOKALIKE_KINDS, UNSEEN_GAPS)

SCAN_POOL = np.array([21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 389, 443, 445, 465, 514,
                      587, 631, 993, 995, 1080, 1433, 1521, 1723, 1883, 2049, 2375, 3000, 3128,
                      3306, 3389, 4369, 5000, 5353, 5432, 5601, 5672, 5900, 5985, 6379, 6443,
                      7001, 8000, 8008, 8080, 8081, 8443, 8888, 9000, 9042, 9090, 9200, 9300,
                      10000, 11211, 15672, 27017, 50000, 50070, 61616])
SERVICE_BY_PORT = {443: 'https', 8443: 'https', 80: 'http', 8080: 'http', 53: 'dns', 22: 'ssh',
                   3389: 'rdp', 25: 'smtp', 587: 'smtp', 465: 'smtp', 21: 'ftp', 123: 'ntp'}
BG_PORTS = np.array([443, 80, 53, 22, 123, 25])
BG_SERVICES = np.array(['https', 'http', 'dns', 'ssh', 'ntp', 'smtp'])
BG_PROTOCOLS = np.array(['tcp', 'tcp', 'udp', 'tcp', 'udp', 'tcp'])
BEACON_PORTS = np.array([8443, 4444, 9001, 8081, 5222])
AUTH_PORTS = np.array([22, 3389])
SCENARIO_BY_COMPONENT = {'scan': 'port_scan', 'bruteforce': 'auth_bruteforce', 'beacon': 'beaconing',
                         'chain_fail': 'auth_bruteforce', 'chain_login': 'auth_bruteforce',
                         'chain_transfer': 'outbound_transfer'}

# Condition parameters. pace=None means familiar 50/50 ordinary/slow.
_BASE = dict(T=1800., rate=(1.6, 2.8), pace=None, attack_lookalike=.3, benign_plain=.4, noisy=False)
CONDITION_PARAMS = {
    'familiar': _BASE,
    'slower': {**_BASE, 'pace': 'slower'},
    'long_sparse': {**_BASE, 'T': 7200.},
    'short': {**_BASE, 'T': 600.},
    'dense': {**_BASE, 'rate': (4., 7.)},
    'sparse': {**_BASE, 'rate': (.5, .9)},
    'noisy_timing': {**_BASE, 'noisy': True},
    'lookalike_heavy': {**_BASE, 'benign_plain': 0.},
    'mixed': {**_BASE, 'benign_plain': 0., 'attack_lookalike': 1.},
}
assert set(CONDITION_PARAMS) == set(CONDITIONS)

_COLUMNS = ('t', 'src_port', 'dst_port', 'service', 'protocol', 'outcome', 'packets_sent', 'packets_recv',
            'bytes_sent', 'bytes_recv', 'duration_s', 'component', 'attack_flow')


def _flows(t, port, outcome, ps, pr, bs, br, dur, component, attack, protocol='tcp', service=None):
    t = np.asarray(t, dtype=float)
    n = len(t)
    port = np.broadcast_to(np.asarray(port, dtype=np.int64), (n,)).copy()
    if service is None:
        service = np.array([SERVICE_BY_PORT.get(int(p), 'other') for p in port], dtype=object)
    full = lambda v, dtype=object: np.broadcast_to(np.asarray(v, dtype=dtype), (n,)).copy()
    return {'t': t, 'src_port': np.zeros(n, dtype=np.int64), 'dst_port': port, 'service': full(service), 'protocol': full(protocol),
            'outcome': full(outcome), 'packets_sent': full(ps, np.int64), 'packets_recv': full(pr, np.int64),
            'bytes_sent': full(bs, np.int64), 'bytes_recv': full(br, np.int64),
            'duration_s': full(dur, float), 'component': full(component), 'attack_flow': full(int(attack), np.int64)}


def _join(parts):
    return {c: np.concatenate([p[c] for p in parts]) for c in _COLUMNS}


def _background(rng, n, T, env_p):
    idx = rng.choice(len(BG_PORTS), n, p=env_p)
    outcome = rng.choice(['established', 'timeout', 'reset', 'rejected'], n, p=[.9, .04, .04, .02])
    failed = np.isin(outcome, ['timeout', 'rejected'])
    ps = np.maximum(1, rng.lognormal(2., 1., n).astype(np.int64))
    pr = np.where(failed, 0, np.maximum(1, (ps * rng.uniform(.5, 1.5, n)).astype(np.int64)))
    udp = BG_PROTOCOLS[idx] == 'udp'
    dur = np.where(udp, rng.lognormal(-3., .5, n), rng.lognormal(.3, 1.2, n))
    return _flows(rng.uniform(0, T, n), BG_PORTS[idx], outcome, ps, pr, ps * rng.integers(60, 1200, n),
                  pr * rng.integers(80, 1400, n), dur, 'background', 0, BG_PROTOCOLS[idx], BG_SERVICES[idx])


def _span(rng, T, pace, ordinary):
    if pace == 'ordinary':
        span = rng.uniform(*ordinary)
    elif pace == 'slow':
        span = rng.uniform(.3, .8) * T
    else:  # 'slower': beyond every training pace
        span = rng.uniform(.85, .98) * T
    span = min(span, .98 * T)
    return rng.uniform(0, T - span), span


def _probe_outcomes(rng, n, failure):
    failed = rng.random(n) < failure
    outcome = np.where(failed, rng.choice(['rejected', 'timeout'], n, p=[.65, .35]),
                       rng.choice(['established', 'reset'], n, p=[.7, .3]))
    ps = rng.integers(1, 3, n)
    pr = np.where(failed, 0, rng.integers(1, 3, n))
    dur = np.where(failed, rng.lognormal(-2.5, .5, n), rng.lognormal(-2., .6, n))
    return outcome, ps, pr, ps * rng.integers(40, 80, n), pr * rng.integers(40, 100, n), dur


def _scan(rng, T, pace):
    n = int(rng.integers(6, 15) if pace == 'slower' else rng.integers(10, 41))
    start, span = _span(rng, T, pace, (20, 180))
    ports = rng.choice(SCAN_POOL, n, replace=False)  # random probe order
    return _flows(start + np.sort(rng.uniform(0, span, n)), ports,
                  *_probe_outcomes(rng, n, rng.uniform(.6, .9)), 'scan', 1)


def _inventory(rng, T):
    n = int(rng.integers(10, 41))
    span = rng.uniform(.2, .9) * T
    ports = np.sort(rng.choice(SCAN_POOL, n, replace=False))  # systematic ascending sweep
    return _flows(rng.uniform(0, T - span) + np.sort(rng.uniform(0, span, n)), ports,
                  *_probe_outcomes(rng, n, rng.uniform(.1, .35)), 'inventory', 0)


def _auth_attempts(rng, n):
    outcome = rng.choice(['reset', 'rejected', 'established'], n, p=[.5, .2, .3])
    ps = rng.integers(4, 10, n)
    pr = np.where(outcome == 'rejected', 0, rng.integers(3, 8, n))
    return outcome, ps, pr, ps * rng.integers(70, 140, n), pr * rng.integers(60, 200, n), rng.uniform(.3, 2.5, n)


def _bruteforce(rng, T, pace):
    n = int(rng.integers(6, 15) if pace == 'slower' else rng.integers(10, 41))
    start, span = _span(rng, T, pace, (30, 300))
    return _flows(start + np.sort(rng.uniform(0, span, n)), rng.choice(AUTH_PORTS),
                  *_auth_attempts(rng, n), 'bruteforce', 1)


def _user_auth(rng, T):
    k = int(rng.integers(1, 5))
    start = rng.uniform(0, T * .9)
    t = np.minimum(start + np.sort(rng.uniform(0, rng.uniform(5, 60), k + 1)), T * .999)
    port = rng.choice(AUTH_PORTS)
    fail = _auth_attempts(rng, k)
    ps = rng.integers(20, 200)
    success = ('established', ps, int(ps * rng.uniform(.8, 1.5)), ps * int(rng.integers(80, 400)),
               ps * int(rng.integers(80, 900)), rng.uniform(60, 1800))
    cols = [np.append(np.asarray(f, dtype=object), s) for f, s in zip(fail, success)]
    return _flows(t, port, cols[0], *[c.astype(float) for c in cols[1:]], 'user_auth', 0)


def _periodic_times(rng, T, interval, jitter, span_fraction, minimum=3):
    span = max(span_fraction * T, minimum * interval * 1.05)
    span = min(span, .98 * T)
    start = rng.uniform(0, T - span)
    steps = interval * np.clip(1 + jitter * rng.standard_normal(int(span / interval) + 2), .2, None)
    times = start + np.cumsum(steps)
    times = times[times < start + span]
    if len(times) < minimum:
        times = start + np.linspace(0, span * .99, minimum)
    return times


def _beacon(rng, T, pace, noisy):
    interval, jitter = {'ordinary': ((30, 90), .1), 'slow': ((150, 400), .25),
                        'slower': ((400, 550), .3)}[pace]
    jitter = min(jitter * (3 if noisy else 1), .9)
    t = _periodic_times(rng, T, rng.uniform(*interval), jitter, rng.uniform(.6, .95))
    if noisy:
        keep = rng.random(len(t)) >= .1
        t = t[keep] if keep.sum() >= 3 else t
    n = len(t)
    sent, recv = rng.uniform(150, 300), rng.uniform(80, 200)
    port = rng.choice(BEACON_PORTS)
    return _flows(t, port, 'established', rng.integers(2, 4, n), rng.integers(1, 3, n),
                  (sent * rng.uniform(.97, 1.03, n)).astype(np.int64),
                  (recv * rng.uniform(.97, 1.03, n)).astype(np.int64), rng.uniform(.05, .3, n), 'beacon', 1)


def _polling(rng, T, noisy):
    jitter = min(rng.uniform(.05, .3) * (3 if noisy else 1), .9)
    t = _periodic_times(rng, T, rng.uniform(30, 400), jitter, rng.uniform(.5, .95))
    n = len(t)
    port = int(rng.choice([443, 80]))
    pr = rng.integers(5, 60, n)
    return _flows(t, port, 'established', rng.integers(3, 9, n), pr, rng.integers(300, 1500, n),
                  rng.lognormal(8.3, .8, n).astype(np.int64) + pr * 40, rng.lognormal(-.5, .6, n), 'polling', 0)


def _retries(rng, T):
    parts = []
    port = int(rng.choice([443, 80, 53]))
    for _ in range(int(rng.integers(1, 4))):
        n = int(rng.integers(3, 11))
        start = rng.uniform(0, T * .95)
        t = np.minimum(start + np.sort(rng.uniform(0, rng.uniform(5, 60), n)), T * .999)
        outcome = rng.choice(['timeout', 'reset'], n, p=[.8, .2])
        ps = rng.integers(1, 4, n)
        parts.append(_flows(t, port, outcome, ps, 0, ps * rng.integers(40, 120, n), 0,
                            np.where(outcome == 'timeout', rng.uniform(3, 30, n), rng.uniform(.01, .2, n)),
                            'retries', 0, 'udp' if port == 53 else 'tcp'))
    return _join(parts)


def _attack(rng, kind, T, pace, noisy):
    if kind == 'scan':
        return _scan(rng, T, pace)
    if kind == 'bruteforce':
        return _bruteforce(rng, T, pace)
    return _beacon(rng, T, pace, noisy)


def _lookalike(rng, kind, T, noisy):
    return {'inventory': lambda: _inventory(rng, T), 'polling': lambda: _polling(rng, T, noisy),
            'retries': lambda: _retries(rng, T), 'user_auth': lambda: _user_auth(rng, T)}[kind]()


def _sorted_window(flows, T, noisy, rng):
    flows['src_port'] = rng.integers(20000, 65535, len(flows['t']))
    t = flows['t']
    if noisy:
        t = np.clip(t + rng.normal(0, 5., len(t)), 0, T * .9999)
    order = np.argsort(t, kind='stable')
    out = {c: v[order] for c, v in flows.items()}
    out['t'] = t[order]
    out['gap'] = np.r_[0., np.diff(out['t'])]
    return out


def general_window(rng, condition, target):
    p = CONDITION_PARAMS[condition]
    T = p['T']
    n_bg = max(3, int(rng.poisson(rng.uniform(*p['rate']) * T / 60)))
    parts = [_background(rng, n_bg, T, rng.dirichlet([8, 3, 4, 1, 1, 1]))]
    key = {'attack_kind': '', 'pace': '', 'lookalike': ''}
    if target:
        key['attack_kind'] = str(rng.choice(ATTACK_KINDS))
        key['pace'] = p['pace'] or str(rng.choice(['ordinary', 'slow']))
        parts.append(_attack(rng, key['attack_kind'], T, key['pace'], p['noisy']))
        if rng.random() < p['attack_lookalike']:
            key['lookalike'] = str(rng.choice(LOOKALIKE_KINDS))
    elif rng.random() >= p['benign_plain']:
        key['lookalike'] = str(rng.choice(LOOKALIKE_KINDS))
    if key['lookalike']:
        parts.append(_lookalike(rng, key['lookalike'], T, p['noisy']))
    return _sorted_window(_join(parts), T, p['noisy'], rng), {**key, 'duration_s': T}


def _gapped(flows, gaps, rng):
    """Each row carries its own gap and ephemeral source port through reordering."""
    flows = dict(flows)
    flows['gap'] = np.asarray(gaps, dtype=float)
    flows['src_port'] = rng.integers(20000, 65535, len(flows['gap']))
    return flows


def chain_triple(rng, gap_choices):
    """Attack, order swap and innocent-early windows sharing context exactly."""
    k = int(rng.integers(6, 15))
    port = int(rng.choice(AUTH_PORTS))
    env_p = rng.dirichlet([8, 3, 4, 1, 1, 1])
    D = int(rng.choice(gap_choices))
    n_pre, n_post = int(rng.integers(3, 11)), int(rng.integers(2, 7))

    def bg(n):
        return _gapped(_background(rng, n, 1., env_p), rng.exponential(25., n), rng)
    pre, post, distract = bg(n_pre), bg(n_post), bg(D)
    decoys = 0
    if D >= 4 and rng.random() < .5:
        decoys = int(rng.integers(2, 4))
        positions = rng.choice(D, decoys, replace=False)
        single = _flows(np.zeros(decoys), port, *_auth_attempts(rng, decoys), 'background', 0)
        single['outcome'] = rng.choice(['reset', 'rejected'], decoys).astype(object)
        for c in _COLUMNS:
            if c not in ('t', 'src_port'):
                distract[c][positions] = single[c]
    fail = _flows(np.zeros(k), port, *_auth_attempts(rng, k), 'chain_fail', 1)
    fail['outcome'] = rng.choice(['reset', 'rejected'], k, p=[.6, .4]).astype(object)
    fail['packets_recv'] = np.where(fail['outcome'] == 'rejected', 0, fail['packets_recv'])
    fail['bytes_recv'] = np.where(fail['outcome'] == 'rejected', 0, fail['bytes_recv'])
    fail = _gapped(fail, rng.uniform(2, 10, k), rng)
    eps = rng.integers(10, 60, k)
    early = _gapped(_flows(np.zeros(k), port, 'established', eps, (eps * rng.uniform(.8, 1.5, k)).astype(np.int64),
                           eps * rng.integers(80, 400, k), eps * rng.integers(80, 900, k),
                           rng.uniform(5, 120, k), 'chain_early', 0), rng.uniform(2, 10, k), rng)
    sps = int(rng.integers(30, 300))
    login = _gapped(_flows([0.], port, 'established', sps, int(sps * rng.uniform(.8, 1.5)),
                           sps * int(rng.integers(80, 400)), sps * int(rng.integers(80, 900)),
                           rng.uniform(60, 900), 'chain_login', 1), [rng.uniform(5, 30)], rng)
    tps = int(rng.integers(2000, 8000))
    tport = int(rng.choice([443, 8443]))
    transfer = _gapped(_flows([0.], tport, 'established', tps, int(tps * rng.uniform(.05, .2)),
                              tps * int(rng.integers(1000, 1400)), tps * int(rng.integers(40, 80)),
                              rng.uniform(20, 300), 'chain_transfer', 1), [rng.uniform(5, 60)], rng)
    st = _cat([login, transfer])
    layouts = {'attack': [pre, fail, distract, st, post], 'swap': [pre, st, distract, fail, post],
               'innocent': [pre, early, distract, st, post]}
    windows = {}
    for role, blocks in layouts.items():
        w = _cat(blocks)
        w['gap'][0] = 0.
        w['t'] = np.cumsum(w['gap'])
        if role != 'attack':
            w['attack_flow'] = np.zeros_like(w['attack_flow'])
            w['component'] = np.where(np.char.startswith(w['component'].astype(str), 'chain_'),
                                      'chain_benign', w['component']).astype(object)
        windows[role] = w
    key = {'gap_distractors': D, 'decoys': decoys, 'chain_k': k}
    return windows, key


def _cat(blocks):
    return {c: np.concatenate([b[c] for b in blocks]) for c in (*_COLUMNS, 'gap')}


def _hex(rng, n=16):
    return ''.join(rng.choice(list('0123456789abcdef'), n))


def build_frame(windows, namespace, seed, rng):
    """windows: list of (flows dict, key dict). Returns (flows DataFrame, key DataFrame)."""
    order = rng.permutation(len(windows))  # window order carries no information
    columns = {c: [] for c in (*_COLUMNS, 'gap', 'session_id', 'event_time', 'flow_id', 'src_host_id')}
    keys, seen = [], set()
    for i in order:
        flows, key = windows[i]
        sid = 'w' + _hex(rng)
        while sid in seen:
            sid = 'w' + _hex(rng)
        seen.add(sid)
        n = len(flows['t'])
        start = float(rng.uniform(0, 1e6))
        for c in (*_COLUMNS, 'gap'):
            columns[c].append(flows[c])
        columns['session_id'].append(np.full(n, sid, dtype=object))
        columns['event_time'].append(start + flows['t'])
        columns['flow_id'].append(np.array([f'{sid}-{j:04d}' for j in range(n)], dtype=object))
        columns['src_host_id'].append(np.full(n, f'{namespace}-h{_hex(rng, 8)}', dtype=object))
        keys.append({**key, 'session_id': sid, 'n_flows': n, 'observation_start_s': start,
                     'observed_span_s': float(flows['t'][-1] - flows['t'][0]),
                     'target': int(flows['attack_flow'].max())})
    data = {c: np.concatenate(v) for c, v in columns.items()}
    frame = pd.DataFrame({
        'schema_version': SCHEMA_VERSION, 'run_id': namespace, 'seed': seed,
        'flow_id': data['flow_id'], 'event_time': data['event_time'], 'session_id': data['session_id'],
        'src_host_id': data['src_host_id'],
        'dst_host_id': [f'{namespace}-svc-{p}' for p in data['dst_port']],
        'src_ip': '10.0.0.1', 'dst_ip': [f'192.0.2.{p % 250 + 1}' for p in data['dst_port']],
        'duration_s': data['duration_s'], 'src_port': data['src_port'],
        'dst_port': data['dst_port'], 'protocol': data['protocol'], 'service': data['service'],
        'packets_sent': data['packets_sent'], 'packets_recv': data['packets_recv'],
        'bytes_sent': data['bytes_sent'], 'bytes_recv': data['bytes_recv'], 'outcome': data['outcome'],
        'prior_gap_s': data['gap'],
        'scenario': [SCENARIO_BY_COMPONENT.get(c, 'benign') if a else 'benign'
                     for c, a in zip(data['component'], data['attack_flow'])],
        'attack_flow': data['attack_flow'], 'component': data['component']})
    frame['label'] = np.where(frame.attack_flow == 1, 'suspicious', 'normal')
    key = pd.DataFrame(keys)
    frame['window_target'] = frame.session_id.map(key.set_index('session_id').target).astype(np.int64)
    validate(frame)
    return frame, key


def generate_split(seed, split, *, windows=0, triples=0, condition='familiar', gaps=FAMILIAR_GAPS,
                   namespace=None):
    """One partition. `split` names the rng stream; balanced targets; triples appended."""
    rng = np.random.default_rng([seed, _stream(split)])
    namespace = namespace or f'sona-eval-{seed}-{split}'
    if windows % 2:
        raise ValueError('Use an even window count for balanced targets.')
    items = []
    targets = rng.permutation(np.tile([0, 1], windows // 2))
    for target in targets:
        flows, key = general_window(rng, condition, int(target))
        items.append((flows, {**key, 'condition': condition, 'triple_id': '', 'role': '',
                              'gap_distractors': -1, 'decoys': 0, 'chain_k': 0}))
    for index in range(triples):
        triple, key = chain_triple(rng, gaps)
        tid = f't{index:05d}-{_hex(rng, 6)}'
        for role, flows in triple.items():
            items.append((flows, {**key, 'condition': 'chain', 'triple_id': tid, 'role': role,
                                  'attack_kind': 'chain' if role == 'attack' else '', 'pace': '',
                                  'lookalike': '', 'duration_s': float(flows['t'][-1]) + 1.}))
    frame, key = build_frame(items, namespace, seed, rng)
    key.insert(0, 'split', split)
    return frame, key


def _stream(split):
    import zlib
    return zlib.crc32(split.encode())


def inference_frame(frame):
    """The only view an adapter receives at inference: no labels, keys or provenance."""
    return frame.loc[:, list(INFERENCE_COLUMNS)].copy()


def window_groups(frame):
    return frame.groupby('session_id', sort=False).indices


def shuffle_interior(frame, seed):
    """Permute all but first/last flow per window; rows carry gaps; times recomputed."""
    rng = np.random.default_rng(seed)
    pieces = []
    for _, positions in window_groups(frame).items():
        idx = positions.copy()
        if len(idx) > 3:
            idx[1:-1] = idx[1:-1][rng.permutation(len(idx) - 2)]
        w = frame.iloc[idx].copy()
        gaps = w.prior_gap_s.to_numpy(copy=True)
        gaps[0] = 0.
        w['prior_gap_s'] = gaps
        w['event_time'] = float(frame.event_time.iloc[positions[0]]) + np.cumsum(gaps)
        pieces.append(w)
    return pd.concat(pieces, ignore_index=True)


def truncate_last(frame, k):
    """Keep only the last k flows of each window; the first kept gap is unknown (0)."""
    pieces = []
    for _, positions in window_groups(frame).items():
        w = frame.iloc[positions[-k:]].copy()
        gaps = w.prior_gap_s.to_numpy(copy=True)
        gaps[0] = 0.
        w['prior_gap_s'] = gaps
        pieces.append(w)
    return pd.concat(pieces, ignore_index=True)


def split_plan(sizes):
    """(split name, kwargs) for every partition of one replicate."""
    plan = [('train', dict(windows=sizes.train_windows, triples=sizes.train_triples)),
            ('val', dict(windows=sizes.val_windows, triples=sizes.val_triples))]
    plan += [(c, dict(windows=sizes.test_windows, condition=c)) for c in CONDITIONS]
    plan += [('chain_familiar', dict(triples=sizes.chain_familiar_triples, gaps=FAMILIAR_GAPS)),
             ('chain_long', dict(triples=sizes.chain_long_triples, gaps=UNSEEN_GAPS))]
    return plan
