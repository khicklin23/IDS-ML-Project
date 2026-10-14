"""Independent suite adapter, with explicit saved versus retrained provenance.

No imports from the evaluator: its test examples remain separate from this code.
"""
import dataclasses
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from .features import CandidatePreprocessor, pack
from .fast_features import FastCandidatePreprocessor
from .models import train, probabilities, make_model
from .inference import CandidatePredictor
from .extended_inference import load_predictor
from .study import Config, targets, write_json
from .retrieval import make_extended
from .parameter_control import train_matched


class EvaluationAdapter:
    causal_features = True

    def __init__(self, architecture='gru', representation='relative', directory=None, mode=None, epochs=80, min_epochs=20,
                 patience=12, batch_size=128, threads=2, max_length=2048, **kwargs):
        if kwargs: raise ValueError(f'Unknown adapter options: {sorted(kwargs)}')
        mode = mode or ('saved' if directory is not None else 'fit')
        if mode not in ('fit', 'saved'): raise ValueError('Mode must be fit or saved.')
        if mode == 'saved' and directory is None: raise ValueError('Saved mode requires a candidate directory.')
        if architecture not in ('gru', 'pool', 'hierarchical', 'retrieval', 'gru40'): raise ValueError('Unsupported evaluation architecture.')
        if representation not in ('raw', 'relative'): raise ValueError('Unsupported representation.')
        self.architecture = architecture
        self.representation = representation
        self.max_length = int(max_length)
        self.config = Config(epochs=int(epochs), min_epochs=int(min_epochs), patience=int(patience),
                             batch_size=int(batch_size), threads=int(threads), max_length=self.max_length)
        self.name = f'candidate_{representation}_{architecture}'
        self.provenance = 'retrained'
        self.processor, self.model, self.artifact_paths, self.history = None, None, [], []
        self.fit_context = None
        if mode == 'saved':
            predictor = load_predictor(directory)
            if predictor.processor is None: raise ValueError('Use neural candidate for this adapter.')
            self.processor, self.model = predictor.processor, predictor.model
            self.architecture = predictor.manifest['architecture']
            self.representation = predictor.manifest['representation']
            self.max_length = predictor.manifest['max_length']
            self.name = f'candidate_saved_{self.architecture}'
            self.provenance = 'saved_checkpoint'
            self.artifact_paths = [str(predictor.directory/f) for f in ('manifest.json', 'model.pt', 'preprocessor.pkl')]

    def fit(self, training, validation, seed):
        if self.provenance == 'saved_checkpoint':
            raise ValueError('Saved transfer adapter cannot be fitted.')
        torch.set_num_threads(self.config.threads)
        self.processor = FastCandidatePreprocessor.fit(training, self.representation)
        tr = pack(training, self.processor, self.max_length)
        va = pack(validation, self.processor, self.max_length)
        self.fit_context = {'seed': int(seed), 'train_windows': len(tr[0]), 'validation_windows': len(va[0])}
        if self.architecture in ('retrieval', 'gru40'):
            self.model, self.history = train_matched((*tr[:2], targets(training)), (*va[:2], targets(validation)),
                self.config, int(seed), lambda dimension: make_extended(self.architecture, dimension))
        else:
            self.model, self.history = train(self.architecture, (*tr[:2], targets(training)),
                                            (*va[:2], targets(validation)), self.config, int(seed))

    def prepare(self, flows):
        if self.processor is None: raise ValueError('Fit or load the adapter first.')
        return pack(flows, self.processor, self.max_length)

    def score_prepared(self, prepared):
        return probabilities(self.model, *prepared[:2])[:, 1].astype(np.float64)

    def window_order(self, prepared):
        return prepared[2].tolist()

    def modules(self):
        return [self.model] if self.model is not None else []

    def flow_features(self, flows):
        if self.processor is None: raise ValueError('Fit or load the adapter first.')
        return self.processor.transform(flows)

    def describe(self):
        return {'name': self.name, 'provenance': self.provenance, 'architecture': self.architecture,
                'representation': self.representation,
                'parameter_count': sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
                'parameter_count_source': 'verified_modules', 'artifact_paths': self.artifact_paths,
                'notes': ['Inputs are prefix-causal; relative elapsed units already contain history when enabled.',
                          'Whole-window classification does not establish online decision latency.',
                          f'Explicit sequence cap {self.max_length}; oversized windows rejected.'],
                'train_config': {key: getattr(self.config, key) for key in ('epochs', 'min_epochs', 'patience', 'batch_size', 'threads', 'max_length')},
                'fit_context': self.fit_context}

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        if self.model is None: raise ValueError('Nothing trained to save.')
        torch.save(self.model.state_dict(), directory/'model.pt')
        (directory/'preprocessor.pkl').write_bytes(pickle.dumps(self.processor))
        record = {'architecture': self.architecture, 'representation': self.representation, 'input_dim': len(self.processor.get_feature_names_out()),
                  'max_length': self.max_length, 'config': dataclasses.asdict(self.config), 'history': self.history,
                  'name': self.name, 'provenance': self.provenance, 'fit_context': self.fit_context}
        write_json(directory/'adapter.json', record)
        self.artifact_paths = [str(directory/f) for f in ('adapter.json', 'model.pt', 'preprocessor.pkl')]

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = json.loads((directory/'adapter.json').read_text())
        self = cls(architecture=record['architecture'], representation=record['representation'])
        self.architecture, self.max_length = record['architecture'], record['max_length']
        self.name, self.provenance = record['name'], record['provenance']
        self.config = Config(**record['config'])
        self.processor = pickle.loads((directory/'preprocessor.pkl').read_bytes())
        factory = make_extended if self.architecture in ('retrieval', 'gru40') else make_model
        self.model = factory(self.architecture, record['input_dim'])
        self.model.load_state_dict(torch.load(directory/'model.pt', weights_only=True, map_location='cpu'))
        self.model.eval(); self.history = record['history']
        self.fit_context = record.get('fit_context')
        self.artifact_paths = [str(directory/f) for f in ('adapter.json', 'model.pt', 'preprocessor.pkl')]
        return self


def build_evaluation_adapter(**kwargs):
    return EvaluationAdapter(**kwargs)
