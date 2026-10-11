"""Deterministic rule-based baseline -- a comparison signal, not a detector.

Per docs/PRODUCT_VISION.md's Detection and replay workspace, the point of a
rule baseline is to give the neural detector something to be measured
against, not to replace it. Do not route alert severity or attack-type
decisions through this module: ids_ml.inference (the model) is the primary
path. This heuristic is intentionally simple and has a known, documented
blind spot (steady "established" traffic such as beaconing) so the
model/rule comparison in evaluate.evaluate_predictions has something real
to show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SUSPICIOUS_OUTCOMES = {"rejected", "timeout"}
_MAX_DURATION_S = 1.0
_MAX_PACKETS_SENT = 3


def rule_score(df: pd.DataFrame) -> pd.DataFrame:
    """Flag short, failed/rejected connection attempts (scan/bruteforce-shaped).

    Known blind spot: beaconing flows look like ordinary short "established"
    connections to this rule, so it will systematically miss them. That gap
    is the point -- it is why the model is the preferred detector.
    """
    out = df.copy()
    is_failed = out["outcome"].astype(str).isin(_SUSPICIOUS_OUTCOMES)
    is_short = out["duration_s"] < _MAX_DURATION_S
    is_sparse = out["packets_sent"] <= _MAX_PACKETS_SENT
    flagged = is_failed & is_short & is_sparse
    out["rule_pred"] = flagged.astype(np.int64)
    return out
