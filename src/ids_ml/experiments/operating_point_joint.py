"""Measure an alert threshold chosen from validation, never from test outcomes.

This diagnostic does not replace native six-class classifications or alert code.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ids_ml.features import load_preprocessor
from ids_ml.models.sessions import build_session_sequences
from ids_ml.models.sequence import SequenceClassifier
from .joint_temporal import sequence_probs
from .temporal_patterns import threshold_for_false_alarms


def measure(directory, target=.01):
    torch.set_num_threads(2)
    metadata = json.loads((directory/'sona_metadata.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    benign = metadata['class_names'].index('benign')
    processor = load_preprocessor(directory/'sona_preprocessor.pkl')
    model = SequenceClassifier(metadata['input_dim'], len(metadata['class_names']), metadata['hidden_dim'])
    model.load_state_dict(torch.load(directory/'sona.pt', weights_only=True))
    frame = pd.read_parquet(directory/'val.parquet')
    X, lengths, y, _ = build_session_sequences(frame, processor, metadata['class_names'],
                                               max_len=metadata['max_seq_len'])
    probabilities = sequence_probs(model, X, lengths)
    scores = 1-probabilities[:, benign]
    threshold = threshold_for_false_alarms(scores, (y!=benign).astype(int), target)
    def metrics(scores, labels, predictions):
        flags = scores >= threshold
        attacks = labels != benign
        return {'sessions': len(labels), 'benign_sessions':int((~attacks).sum()),
                'attack_sessions':int(attacks.sum()),
                'false_alerts':int((flags & ~attacks).sum()),
                'missed_attacks':int((~flags & attacks).sum()),
                'false_alert_rate':float(flags[~attacks].mean()) if (~attacks).any() else None,
                'attack_recall':float(flags[attacks].mean()) if attacks.any() else None,
                'native_classification_accuracy':float(np.mean(predictions==labels))}
    result = {'validation_false_alert_target':target, 'threshold':threshold,
              'score':'1 - P(benign)', 'validation':metrics(scores,y,probabilities.argmax(axis=1)),
              'tests':{}, 'note':'Threshold chosen exclusively on validation benign sessions. Native labels are unchanged; thresholding cannot repair a wrong attack-family label.'}
    extra_path=directory/'extra_stress/report.json'
    extra=json.loads(extra_path.read_text()) if extra_path.exists() else None
    splits=list(report['results']) + (list(extra['results']) if extra else [])
    for split in splits:
        scores_directory=directory/'extra_stress' if extra and split in extra['results'] else directory
        with np.load(scores_directory/f'{split}_predictions.npz') as saved:
            result['tests'][split] = metrics(1-saved['sona_probs'][:,benign],saved['y'],saved['sona'])
    (directory/'alert_operating_point.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--target',type=float,default=.01)
    args=parser.parse_args()
    print(json.dumps(measure(args.directory,args.target),indent=2))
