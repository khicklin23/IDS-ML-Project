"""Unified local inference for original and retrieval-extension checkpoints."""
import json
import pickle
from pathlib import Path
import torch
from ids_ml.experiments.mixed_scan import digest
from .features import pack
from .models import probabilities
from .retrieval import make_extended
from .inference import CandidatePredictor


class ExtendedPredictor:
    @classmethod
    def load(cls, directory):
        obj = cls(); obj.directory = Path(directory)
        obj.manifest = json.loads((obj.directory/'manifest.json').read_text())
        if obj.manifest.get('artifact_version') != 'sona-retrieval-1.0': raise ValueError('Unsupported extension artifact.')
        for filename, expected in obj.manifest['sha256'].items():
            if digest(obj.directory/filename) != expected: raise ValueError(f'Artifact changed: {filename}')
        obj.processor = pickle.loads((obj.directory/'preprocessor.pkl').read_bytes())
        obj.model = make_extended(obj.manifest['architecture'], obj.manifest['input_dim'])
        obj.model.load_state_dict(torch.load(obj.directory/'model.pt', map_location='cpu', weights_only=True))
        obj.model.eval(); return obj

    def predict(self, frame):
        x, lengths, ids, _ = pack(frame, self.processor, self.manifest['max_length'])
        probs = probabilities(self.model, x, lengths)
        return {'session_ids': ids, 'probabilities': probs, 'scores': probs[:, 1].astype('float64')}


def load_predictor(directory):
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    if manifest['architecture'] in ('retrieval', 'gru40'): return ExtendedPredictor.load(directory)
    return CandidatePredictor.load(directory)
