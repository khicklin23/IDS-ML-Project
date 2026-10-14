"""Inference-only optimization of the frozen features, preserving fitted values.

Avoid repeated DataFrame slicing and repeated validation. No feature or model
change: packaging requires exact transformed-array and prediction equivalence.
"""
import numpy as np
from ids_ml.data.schema import FEATURE_COLUMNS
from ids_ml.experiments.scan_context import PACE_COLUMNS
from .features import CandidatePreprocessor


class FastCandidatePreprocessor(CandidatePreprocessor):
    version = 'sona-candidate-relative-fast-1.0'

    @classmethod
    def from_processor(cls, processor):
        return cls(processor.encoder, processor.mode)

    def transform(self, frame):
        if len(frame) == 0 or frame.session_id.isna().any():
            raise ValueError('Nonempty telemetry with session identities is required.')
        times = frame.event_time.to_numpy(dtype=float)
        gaps = frame.prior_gap_s.to_numpy(dtype=float)
        if not np.isfinite(times).all() or not np.isfinite(gaps).all():
            raise ValueError('Finite times and gaps required.')
        timing = np.zeros((len(frame), 2), dtype=np.float64) if self.mode == 'relative' else None
        groups = frame.groupby('session_id', sort=False).indices.values()
        for rows in groups:
            t, g = times[rows], gaps[rows]
            diff = np.diff(t)
            if np.any(diff < 0) or not np.allclose(g, np.r_[0., diff], atol=1e-7, rtol=1e-7):
                raise ValueError('Chronological sessions and consistent gaps required.')
            if timing is None: continue
            sums = np.r_[0., np.cumsum(np.where(g > 0, g, 0.))[:-1]]
            counts = np.r_[0, np.cumsum(g > 0)[:-1]]
            scale = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            valid = scale > 0
            timing[rows[valid], 0] = np.log1p(g[valid]/scale[valid])
            timing[rows[valid], 1] = np.log1p((t[valid]-t[0])/scale[valid])
        prepared = frame[list(FEATURE_COLUMNS)].copy()
        if timing is not None: prepared[list(PACE_COLUMNS)] = timing
        return self.encoder.transform(prepared).astype(np.float32)
