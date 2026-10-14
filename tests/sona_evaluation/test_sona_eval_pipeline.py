"""End-to-end smoke run: generation, fitting, frozen thresholds, scoring, audit, resume, summary."""
import json

import numpy as np
import pytest

from ids_ml.experiments.sona_evaluation import run, summarize
from ids_ml.experiments.sona_evaluation.metrics import threshold_at_budget


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp('sona_eval') / 'smoke'
    run.main(['--out-dir', str(out), '--mode', 'smoke',
              '--skip-models', 'beta1_saved', 'beta2_saved', 'sona_shuffled_trained', 'summary_boosting'])
    return out


def test_smoke_run_passes_every_audit(smoke):
    rep = smoke / 'replicate-52001'
    report = json.loads((rep / 'audit.json').read_text())
    assert report['status'] == 'passed', {k: v for k, v in report['checks'].items() if not v['passed']}
    assert json.loads((smoke / 'run_audit.json').read_text()) == {'public_artifacts_unchanged': True,
                                                                 'suite_source_unchanged': True}


def test_thresholds_were_frozen_from_validation_scores(smoke):
    rep = smoke / 'replicate-52001'
    frozen = json.loads((rep / 'thresholds.json').read_text())['thresholds']
    import pandas as pd
    key = pd.read_parquet(rep / 'data' / 'val_key.parquet')
    with np.load(rep / 'scores' / 'val.npz') as val:
        for model, budgets in frozen.items():
            for budget, threshold in budgets.items():
                assert threshold == threshold_at_budget(val[model], key.target.to_numpy(), float(budget))


def test_resume_is_a_no_op_and_source_changes_are_refused(smoke):
    rep = smoke / 'replicate-52001'
    before = {p.name: p.stat().st_mtime for p in (rep / 'scores').glob('*.npz')}
    run.main(['--out-dir', str(smoke), '--mode', 'smoke',
              '--skip-models', 'beta1_saved', 'beta2_saved', 'sona_shuffled_trained', 'summary_boosting'])
    assert before == {p.name: p.stat().st_mtime for p in (rep / 'scores').glob('*.npz')}
    stamp = json.loads((smoke / 'protocol.json').read_text())
    stamp['suite_source_sha256']['cases.py'] = 'changed'
    (smoke / 'protocol.json').write_text(json.dumps(stamp))
    with pytest.raises(SystemExit):
        run.main(['--out-dir', str(smoke), '--mode', 'smoke'])


def test_summary_reports_every_condition_budget_and_memory_set(smoke):
    summarize.main([str(smoke), '--compare', 'sona_counterpart', 'beta2_counterpart'])
    summary = json.loads((smoke / 'summary.json').read_text())
    assert set(summary['detection']) == set(run.CONDITIONS)
    row = summary['detection']['familiar']['sona_counterpart']
    assert set(row) >= {'0.01', '0.02', '0.05', 'roc_auc'}
    assert {'chain_familiar', 'chain_long', 'chain_familiar:shuffled'} <= set(summary['memory'])
    comp = summary['comparisons']['sona_counterpart - beta2_counterpart']
    assert len(comp['conditions']) == len(run.CONDITIONS) and 'chain_long:triple_solved' in comp
    assert (smoke / 'RESULTS.md').read_text().startswith('# Sona evaluation results')
