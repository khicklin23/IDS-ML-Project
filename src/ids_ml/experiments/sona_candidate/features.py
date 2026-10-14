"""Explicit causal timing inference; no optional caller-prepared feature path."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ids_ml.data.schema import FEATURE_COLUMNS
from ids_ml.experiments.scan_context import CATEGORICAL, PACE_COLUMNS


def prepare(frame, mode='relative'):
    """Vectorized exact equivalent of the frozen relative-only timing features."""
    validate_input(frame)
    result = frame[list(FEATURE_COLUMNS)].copy()
    if mode == 'raw': return result
    timing = np.zeros((len(frame), 2), dtype=np.float64)
    for rows in frame.groupby('session_id', sort=False).indices.values():
        gaps = frame.prior_gap_s.iloc[rows].to_numpy(dtype=float)
        times = frame.event_time.iloc[rows].to_numpy(dtype=float)
        past_sum = np.r_[0., np.cumsum(np.where(gaps > 0, gaps, 0.))[:-1]]
        past_count = np.r_[0, np.cumsum(gaps > 0)[:-1]]
        scale = np.divide(past_sum, past_count, out=np.zeros_like(past_sum), where=past_count > 0)
        valid = scale > 0
        timing[rows[valid], 0] = np.log1p(gaps[valid] / scale[valid])
        timing[rows[valid], 1] = np.log1p((times[valid]-times[0]) / scale[valid])
    result[list(PACE_COLUMNS)] = timing
    return result


class CandidatePreprocessor:
    version = 'sona-candidate-relative-1.0'

    def __init__(self, encoder, mode):
        self.encoder, self.mode = encoder, mode

    @classmethod
    def fit(cls, frame, mode='relative'):
        if mode not in ('raw', 'relative'):
            raise ValueError('Only raw and relative representations are supported.')
        columns = list(FEATURE_COLUMNS)
        if mode == 'relative':
            columns.remove('prior_gap_s')
            columns += list(PACE_COLUMNS)
        encoder = ColumnTransformer([
            ('numeric', StandardScaler(), [c for c in columns if c not in CATEGORICAL]),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL))])
        encoder.fit(prepare(frame, mode)[columns])
        return cls(encoder, mode)

    def transform(self, frame):
        validate_input(frame)
        return self.encoder.transform(prepare(frame, self.mode)).astype(np.float32)

    def get_feature_names_out(self):
        return self.encoder.get_feature_names_out()


def validate_input(frame):
    if len(frame) == 0 or frame.session_id.isna().any():
        raise ValueError('Nonempty telemetry with session identities is required.')
    for rows in frame.groupby('session_id', sort=False).indices.values():
        times = frame.event_time.iloc[rows].to_numpy(dtype=float)
        gaps = frame.prior_gap_s.iloc[rows].to_numpy(dtype=float)
        expected = np.r_[0., np.diff(times)]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError('Sessions must contain finite chronological timestamps.')
        if not np.isfinite(gaps).all() or not np.allclose(gaps, expected, atol=1e-7, rtol=1e-7):
            raise ValueError('Gaps must match successive observations, starting at zero.')


def pack(frame, processor, cap=160):
    features = processor.transform(frame)
    groups = list(frame.groupby('session_id', sort=False).indices.values())
    lengths = np.array([len(g) for g in groups], dtype=np.int64)
    if np.any(lengths > cap):
        raise ValueError('Sequence exceeds artifact cap; refusing silent truncation.')
    x = np.zeros((len(groups), int(lengths.max()), features.shape[-1]), dtype=np.float32)
    for i, rows in enumerate(groups):
        x[i, :len(rows)] = features[rows]
    ids = frame.groupby('session_id', sort=False).session_id.first().to_numpy(dtype=str)
    return x, lengths, ids, groups
