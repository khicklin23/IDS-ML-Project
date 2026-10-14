"""Causal multiscale event states plus two shared learned memory retrieval hops."""
import torch
from torch import nn
from ids_ml.models.sequence import SequenceClassifier


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2*dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x):
        past = self.depthwise(nn.functional.pad(x, (self.padding, 0)))
        return x + torch.tanh(self.mix(past))


class RetrievalMemory(nn.Module):
    """Retrieve relevant event states twice; whole-window classification only."""
    def __init__(self, dimension, classes=2):
        super().__init__()
        self.project = nn.Linear(dimension, 32)
        self.blocks = nn.Sequential(*[CausalBlock(32, dilation) for dilation in (1, 4, 16)])
        self.query = nn.Linear(32, 32)
        self.energy = nn.Linear(32, 1, bias=False)
        self.update = nn.Linear(64, 32)
        self.head = nn.Linear(32, classes)

    def encode(self, x):
        return self.blocks(torch.tanh(self.project(x)).transpose(1, 2)).transpose(1, 2)

    def forward(self, x, lengths):
        states = self.encode(x)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        query = states[torch.arange(len(x), device=x.device), lengths-1]
        for _ in range(2):
            energy = self.energy(torch.tanh(states + self.query(query)[:, None, :])).squeeze(-1)
            weights = torch.softmax(energy.masked_fill(~valid, -torch.inf), 1)
            retrieved = (weights[..., None]*states).sum(1)
            query = query + torch.tanh(self.update(torch.cat((query, retrieved), -1)))
        return self.head(query)


class ExtendedGRU(SequenceClassifier):
    def __init__(self, dimension):
        super().__init__(dimension, 2, 40)


def make_extended(architecture, dimension):
    if architecture == 'retrieval': return RetrievalMemory(dimension)
    if architecture == 'gru40': return ExtendedGRU(dimension)
    raise ValueError('Unknown extended architecture.')
