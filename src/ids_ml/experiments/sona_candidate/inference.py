"""Stable independent-evaluation adapter for trusted local candidate artifacts."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from ids_ml.experiments.mixed_scan import summaries, digest
from .features import pack, validate_input
from .models import make_model, probabilities


class CandidatePredictor:
    @classmethod
    def load(cls, directory):
        obj = cls()
        obj.directory = Path(directory)
        obj.manifest = json.loads((obj.directory/'manifest.json').read_text())
        if obj.manifest['artifact_version'] != 'sona-candidate-1.0':
            raise ValueError('Unsupported candidate artifact.')
        for filename, expected in obj.manifest['sha256'].items():
            if digest(obj.directory/filename) != expected:
                raise ValueError(f'Artifact hash mismatch: {filename}')
        if obj.manifest['architecture'] == 'summary':
            obj.model = pickle.loads((obj.directory/'model.pkl').read_bytes())
            obj.processor = None
        else:
            obj.processor = pickle.loads((obj.directory/'preprocessor.pkl').read_bytes())
            obj.model = make_model(obj.manifest['architecture'], obj.manifest['input_dim'])
            obj.model.load_state_dict(torch.load(obj.directory/'model.pt', map_location='cpu', weights_only=True))
            obj.model.eval()
        return obj

    def predict(self, frame):
        validate_input(frame)
        if self.processor is None:
            groups = list(frame.groupby('session_id', sort=False).indices.values())
            if max(map(len, groups)) > self.manifest['max_length']:
                raise ValueError('Sequence exceeds artifact cap.')
            values, _ = summaries(frame, groups)
            probs = self.model.predict_proba(values)
            ids = frame.groupby('session_id', sort=False).session_id.first().to_numpy(dtype=str)
        else:
            x, lengths, ids, _ = pack(frame, self.processor, self.manifest['max_length'])
            probs = probabilities(self.model, x, lengths)
        return {'session_ids': ids, 'probabilities': probs, 'scores': probs[:, 1].astype(np.float64)}
