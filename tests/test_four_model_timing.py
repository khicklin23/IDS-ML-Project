"""Regression for names reused across independent generation runs."""
import pandas as pd
from ids_ml.experiments.four_model_timing import session_keys


def test_split_identity_includes_run_and_normalizes_combined_session_names():
    ordinary = pd.DataFrame({'run_id':['run-701'], 'session_id':['host-0001']})
    stealth = pd.DataFrame({'run_id':['run-1701'], 'session_id':['host-0001']})
    combined = pd.DataFrame({'run_id':['run-701'], 'session_id':['run-701::host-0001']})
    assert not session_keys(ordinary) & session_keys(stealth)
    # A real repeated session in the same run must still be caught after namespacing.
    assert session_keys(ordinary) & session_keys(combined)
