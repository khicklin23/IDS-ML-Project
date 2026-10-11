"""Publish audited matched-study summaries without changing public model reports."""
import argparse
import json
from pathlib import Path


def sync(root, destination, four_model_summary=None):
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['seeds'] == [701, 733, 769]
    assert len(summary['records']) == 3
    for record in summary['records']:
        assert record['matched_raw_datasets'] == 'passed'
        for variant in record['variants'].values():
            assert variant['extra']['audit_status'] == 'passed'
    aggregates = summary['aggregate']

    def score(mode, test, model):
        values = [r for r in aggregates if (r['mode'], r['test'], r['model']) == (mode, test, model)]
        if not values:
            return None
        assert len(values) == 1
        value = values[0]
        return {'mean': value['mean_accuracy'], 'min': value['min_accuracy'], 'max': value['max_accuracy']}

    tests = [('timing', 'Same flows, different order'), ('extra_noise', 'Heavier timing noise'),
             ('extreme_noise', 'Noise beyond training'), ('slow_timing', 'Slower, noisier cycles'),
             ('unseen_long_cycles', 'New cycle lengths'), ('subtle_timing', 'Small timing differences'),
             ('extreme_pace', 'Much slower paired cycles'),
             ('unseen_polling', 'Harmless polling: 11–17 minute gaps'),
             ('extreme_polling', 'Harmless polling: 25–50 minute gaps')]
    models = [{'id': 'beta2', 'label': 'Beta 2'}, {'id': 'hybrid', 'label': 'Sona · seconds'},
              {'id': 'hybrid_log', 'label': 'Sona · compressed'}, {'id': 'relative', 'label': 'Sona · relative'},
              {'id': 'timing_summary', 'label': 'Timing summaries'}]
    rows = []
    for test, label in tests:
        beta = score('relative', test, 'beta2')
        sona = score('hybrid_log', test, 'sona_hybrid')
        scores = {'beta2': beta, 'timing_summary': score('hybrid_log', test, 'timing_summary')}
        scores.update({mode: score(mode, test, 'sona_hybrid') for mode in ('hybrid', 'hybrid_log', 'relative')})
        # The summary baseline reads the same raw timing data in every preparation.
        for mode in ('hybrid', 'relative'):
            assert score(mode, test, 'timing_summary') == scores['timing_summary']
        sessions = next(r['sessions'] for r in aggregates if (r['mode'], r['test'], r['model']) == ('hybrid_log', test, 'sona_hybrid'))
        rows.append({'label': label, 'beta2': beta['mean'], 'sona': sona['mean'],
                     'sona_min': sona['min'], 'sona_max': sona['max'], 'sessions': sessions, 'scores': scores})
    gates = summary['candidate_gate']
    failures = [models[i+1]['label'] for i, mode in enumerate(('hybrid', 'hybrid_log', 'relative')) if not gates[mode]['eligible']]
    selection = summary['selected_mode']
    conclusion = ('No preparation passed every predeclared check.' if selection is None else
                  f"The {next(m['label'] for m in models if m['id'] == selection)} preparation passed the predeclared showcase checks.")
    if failures:
        conclusion += ' Did not pass all checks: ' + ', '.join(failures) + '.'
    study = {'title': 'Pace and noise: matched follow-up',
        'explanation': 'Nine Sona candidates were trained on three independently generated datasets. Each preparation receives the same traffic and training settings. Seconds keeps the original gap. Compressed makes large gaps less extreme. Relative compares each gap with the session’s average pace and removes absolute speed. These are class guesses, not threshold-based alert scores. Higher scores mean more correct session guesses; on harmless polling, they mean fewer false alerts. Polling ranges describe the configured base pace; individual gaps vary. Small numbers below each score show the range across three runs.',
        'generated_flows': summary['unique_flows'], 'seeds': summary['seeds'], 'models': models, 'rows': rows,
        'benign_controls': [], 'limit': conclusion + ' These candidates have not replaced the public model.',
        'provenance': 'Source: sona_pace_noise_v1. Counts include unique generated data once, not copies processed by nine candidates. Beta 2 uses the relative-input runs, matching the eligible relative-pace candidate; each variant’s matching Beta 2 results are retained in the source CSV. Timing summaries is a simple classifier using predefined sequence statistics, trained only on the binary timing subset; it has no six-class scores. More noise was included in training, but the strongest noise and very slow harmless polling tests remain outside the training range. Saved predictions, matched splits and shuffled-sequence controls were audited. All candidates meet the ordinary and less-obvious traffic check, but those tests alone do not establish an advantage over Beta 2. All results use synthetic, completed sessions.'}
    study['comparison'] = {
        'mode': 'relative',
        'ordinary_sessions': next(r['sessions'] for r in aggregates if (r['mode'], r['test'], r['model']) == ('relative', 'normal', 'sona_hybrid')),
        'timing_sessions': next(r['sessions'] for r in aggregates if (r['mode'], r['test'], r['model']) == ('relative', 'timing', 'sona_hybrid')),
        'models': {label: {'ordinary': score('relative', 'normal', model)['mean'],
                           'timing': score('relative', 'timing', model)['mean']}
                   for label, model in [('beta2', 'beta2'), ('sona', 'sona_hybrid')]}
    }
    existing = json.loads(destination.read_text())
    archive = [dict(item, archived=True) for item in existing['studies'] if item['title'] != study['title']]
    existing['studies'] = [study, *archive]
    if four_model_summary is not None:
        four = json.loads(four_model_summary.read_text())
        assert four['schema'] == 1 and four['audit_status'] == 'passed' and four['seeds'] == summary['seeds']
        assert all(record['audit']['status'] == 'passed' for record in four['records'])
        measurements = {}
        for version in ('v1', 'v2', 'v3', 'v4'):
            measurements[version] = {test: next(record for record in four['aggregate']
                if (record['version'], record['test']) == (version, test))
                for test in four['tests']}
            expected = sum(measurements[version][test]['mean'] for test in ('normal', 'stealth', 'timing')) / 3
            assert abs(expected - four['three_test_scores'][version]['mean']) < 1e-12
        for version, model in [('v2', 'beta2'), ('v4', 'sona')]:
            assert measurements[version]['normal']['mean'] == study['comparison']['models'][model]['ordinary']
            assert measurements[version]['timing']['mean'] == study['comparison']['models'][model]['timing']
        existing['four_model_comparison'] = {'schema': 1, 'seeds': four['seeds'], 'results': measurements,
            'source': 'four_model_timing_v1',
            'scope': 'All four candidates evaluated on identical saved test sessions. Timing training is shared; Beta 1/Beta 3 exclude slowed-down ordinary attack training. These are retrained counterparts, not original public beta checkpoints. Complete sessions only; speed was not measured.'}

    destination.write_text(json.dumps(existing, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1] / 'research.json')
    parser.add_argument('--four-model-summary', type=Path)
    args = parser.parse_args()
    sync(args.root, args.destination, args.four_model_summary)
