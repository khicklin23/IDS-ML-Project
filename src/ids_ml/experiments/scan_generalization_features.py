"""Independent duration/density conditions and causal feature ablations."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ids_ml.data.schema import FEATURE_COLUMNS
from .scan_context import (generate_context_windows, causal_features, ContextPreprocessor,
                           CATEGORICAL, PACE_COLUMNS)

SMALL_HISTORY = ('recent32_failed_fraction', 'recent32_unique_failed_ports', 'current_port_is_new')
CONDITIONS = ('test', 'duration', 'density', 'combined', 'inventory')

def generate_condition(seed, count, prefix, condition):
    if condition not in CONDITIONS: raise ValueError('Unknown condition.')
    mode = 'lookalikes' if condition == 'inventory' else ('shifted' if condition in ('density', 'combined') else 'usual')
    frame = generate_context_windows(seed, count, prefix, mode=mode)
    duration = 3600. if condition in ('duration', 'combined', 'inventory') else 1800.
    factor = duration / float(frame.observation_duration_s.iloc[0])
    frame['event_time'] *= factor
    frame['observation_start_s'] *= factor
    frame['observation_duration_s'] = duration
    frame['prior_gap_s'] = frame.groupby('session_id').event_time.diff().fillna(0.)
    frame['condition'] = condition
    return frame

def generate_partition(seed, count, prefix, condition):
    if condition != 'mixture': return generate_condition(seed, count, prefix, condition)
    if count % 8: raise ValueError('Mixture count must be divisible by eight.')
    return pd.concat([generate_condition(seed + i * 1000, count // 4,
        f'{prefix}-{name}', name) for i, name in enumerate(CONDITIONS[:4])], ignore_index=True)

def prepare(frame):
    result = causal_features(frame)
    # Scale-free startup: no prior positive gap means timing has no established reference.
    # After the first positive gap, denominators use only strictly prior positive gaps.
    timing = np.zeros((len(frame), 2), dtype=float)
    for positions in frame.groupby('session_id', sort=False).indices.values():
        gaps = frame.prior_gap_s.iloc[positions].to_numpy(dtype=float)
        times = frame.event_time.iloc[positions].to_numpy(dtype=float)
        total, count = 0., 0
        for i, pos in enumerate(positions):
            scale = total / count if count else None
            timing[pos] = ([np.log1p(gaps[i] / scale),
                np.log1p((times[i] - times[0]) / scale)] if scale else [0., 0.])
            if gaps[i] > 0: total += gaps[i]; count += 1
    result[list(PACE_COLUMNS)] = timing
    return result

def build_processor(train, mode, prepared=None):
    if mode not in ('raw', 'relative', 'small'): raise ValueError('Unknown representation.')
    prepared = prepare(train) if prepared is None else prepared
    columns = list(FEATURE_COLUMNS)
    if mode != 'raw':
        columns.remove('prior_gap_s')
        columns += list(PACE_COLUMNS)
    if mode == 'small': columns += list(SMALL_HISTORY)
    encoder = ColumnTransformer([
        ('numeric', StandardScaler(), [c for c in columns if c not in CATEGORICAL]),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL))])
    encoder.fit(prepared[columns])
    return ContextPreprocessor(encoder, mode)
