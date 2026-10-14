"""Causal telemetry context for the separate mixed-scan study."""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ids_ml.data.schema import FEATURE_COLUMNS

PACE_COLUMNS = ('online_relative_gap', 'online_elapsed_units')
HISTORY_COLUMNS = ('past_flows', 'past_failed', 'past_unique_ports', 'past_unique_failed_ports',
                   'recent8_failed_fraction', 'recent8_unique_ports', 'recent8_unique_failed_ports',
                   'recent32_failed_fraction', 'recent32_unique_ports', 'recent32_unique_failed_ports',
                   'past300_failed', 'past300_unique_failed_ports', 'current_port_is_new')
CATEGORICAL = ('protocol', 'service', 'outcome')


def generate_context_windows(seed, count, prefix, *, mode='usual'):
    """Apply an observer's fixed capture boundary uniformly to all activity.

    The original simulator occasionally schedules late inventory flows beyond
    the nominal duration. Preserve it for v1 reproduction, but do not include
    out-of-window observations in this new study.
    """
    from .mixed_scan import generate_windows
    frame = generate_windows(seed, count, prefix, mode=mode)
    index = frame.session_id.str.rsplit('-', n=1).str[-1].astype(int)
    frame['observation_start_s'] = (index // 40) * frame.observation_duration_s
    keep = (frame.event_time >= frame.observation_start_s) & (frame.event_time < frame.observation_start_s + frame.observation_duration_s)
    frame = frame.loc[keep].copy().reset_index(drop=True)
    frame['prior_gap_s'] = frame.groupby('session_id').event_time.diff().fillna(0.)
    return frame


def causal_features(frame):
    """Current flow plus strictly earlier observations in its supplied window.

    No full-window normalization, labels, behavior, identities-as-values, or
    future event measurements. Initial pace fallback is a fixed 30 seconds.
    """
    result = frame[list(FEATURE_COLUMNS)].copy()
    extra = np.zeros((len(frame), len(PACE_COLUMNS) + len(HISTORY_COLUMNS)), dtype=np.float64)
    for positions in frame.groupby('session_id', sort=False).indices.values():
        w = frame.iloc[positions]
        times = w.event_time.to_numpy(dtype=float)
        if np.any(np.diff(times) < 0):raise ValueError('Context requires chronological windows.')
        ports = w.dst_port.to_numpy()
        failures = w.outcome.isin(['rejected', 'timeout']).to_numpy()
        gaps = w.prior_gap_s.to_numpy(dtype=float)
        if not np.isfinite(gaps).all() or np.any(gaps < 0):raise ValueError('Invalid gaps.')
        seen, failed_seen = set(), set()
        recent8, recent32, recent300 = deque(), deque(), deque()
        pace_sum, pace_count, failure_count = 0., 0, 0
        def recent_features(queue):
            return (sum(f for _, f in queue) / max(len(queue), 1),
                    len({p for p, _ in queue}),len({p for p, f in queue if f}))
        for i, pos in enumerate(positions):
            time_s, port, failed, gap = times[i], int(ports[i]), bool(failures[i]), gaps[i]
            scale = pace_sum / pace_count if pace_count else 30.
            while recent300 and recent300[0][0] < time_s - 300:recent300.popleft()
            extra[pos] = [np.log1p(gap / max(scale, 1e-6)),
                          np.log1p((time_s - times[0]) / max(scale, 1e-6)),
                          i, failure_count,len(seen),len(failed_seen),
                          *recent_features(recent8),*recent_features(recent32),
                          sum(f for _, _, f in recent300),
                          len({p for _, p, f in recent300 if f}),int(port not in seen)]
            # Update only after extracting features for this observation.
            seen.add(port)
            if failed:failed_seen.add(port);failure_count += 1
            recent8.append((port,failed));recent32.append((port,failed));recent300.append((time_s,port,failed))
            if len(recent8)>8:recent8.popleft()
            if len(recent32)>32:recent32.popleft()
            if gap>0:pace_sum += gap;pace_count += 1
    result[list(PACE_COLUMNS + HISTORY_COLUMNS)] = extra
    return result


class ContextPreprocessor:
    def __init__(self, encoder, mode):
        self.encoder, self.mode = encoder, mode

    def transform(self, frame, prepared=None):
        prepared = causal_features(frame) if prepared is None else prepared
        if len(prepared) != len(frame):raise ValueError('Prepared rows do not match telemetry.')
        return self.encoder.transform(prepared).astype(np.float32)

    def get_feature_names_out(self):return self.encoder.get_feature_names_out()


def build_context_preprocessor(train, mode='context', prepared=None):
    if mode not in ('raw','pace','context'):raise ValueError('Unknown context mode.')
    prepared = causal_features(train) if prepared is None else prepared
    columns = list(FEATURE_COLUMNS)
    if mode in ('pace','context'):columns += list(PACE_COLUMNS)
    if mode=='context':columns += list(HISTORY_COLUMNS)
    encoder = ColumnTransformer([
        ('numeric', StandardScaler(), [c for c in columns if c not in CATEGORICAL]),
        ('categorical', OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(CATEGORICAL))])
    encoder.fit(prepared[columns])
    return ContextPreprocessor(encoder,mode)


def pack_context(frame, processor, cap, prepared=None):
    features = processor.transform(frame,prepared)
    groups = list(frame.groupby('session_id',sort=False).indices.values())
    if max(map(len,groups))>cap:raise ValueError('Window exceeds fixed cap.')
    X=np.zeros((len(groups),cap,features.shape[-1]),dtype=np.float32)
    lengths=np.array(list(map(len,groups)),dtype=np.int64)
    y=np.array([frame.window_target.iloc[g[0]] for g in groups],dtype=np.int64)
    for i,g in enumerate(groups):
        if frame.window_target.iloc[g].nunique()!=1:raise ValueError('Inconsistent targets.')
        X[i,:len(g)]=features[g]
    return X,lengths,y,groups


def context_summaries(prepared, groups):
    columns=list(PACE_COLUMNS+HISTORY_COLUMNS)
    a=prepared[columns].to_numpy(dtype=float)
    return np.array([np.r_[a[g].mean(axis=0),a[g].max(axis=0),a[g[-1]]] for g in groups],dtype=np.float32)
