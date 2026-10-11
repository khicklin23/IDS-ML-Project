"""Small PyTorch binary classifier: the Foundation-stage baseline detector."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """A deliberately small MLP: baseline to compare future complexity against.

    output_dim=1 (default) is the binary normal/suspicious head, trained with
    BCEWithLogitsLoss. output_dim>1 is a multiclass attack-type head (benign +
    each attack family), trained with CrossEntropyLoss -- see
    ids_ml.models.train.train_multiclass_model. Same small architecture either
    way; only the output width and loss differ.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.output_dim == 1 else out  # raw logits either way
