"""Small memory controls and a bounded hierarchical memory alternative."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ids_ml.models.sequence import SequenceClassifier


class PoolClassifier(nn.Module):
    def __init__(self, dimension, classes=2):
        super().__init__()
        self.flow = nn.Sequential(nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, classes))

    def forward(self, x, lengths):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        return (self.flow(x) * mask[..., None]).sum(1) / lengths[:, None]


class HierarchicalMemory(nn.Module):
    def __init__(self, dimension, classes=2, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.project = nn.Sequential(nn.Linear(dimension, 16), nn.Tanh())
        self.gru = nn.GRU(32, 24, batch_first=True)
        self.attention = nn.Linear(24, 1)
        self.head = nn.Linear(48, classes)

    def forward(self, x, lengths):
        b, n, _ = x.shape
        blocks = (n + self.block_size - 1) // self.block_size
        z = self.project(x)
        mask = torch.arange(n, device=x.device)[None, :] < lengths[:, None]
        padding = blocks * self.block_size - n
        z = nn.functional.pad(z, (0, 0, 0, padding))
        mask = nn.functional.pad(mask, (0, padding))
        z = z.reshape(b, blocks, self.block_size, 16)
        mask = mask.reshape(b, blocks, self.block_size)
        mean = (z * mask[..., None]).sum(2) / mask.sum(2).clamp(min=1)[..., None]
        maximum = z.masked_fill(~mask[..., None], -torch.inf).max(2).values
        maximum = maximum.masked_fill(~mask.any(2)[..., None], 0.)
        block_lengths = (lengths + self.block_size - 1) // self.block_size
        packed = pack_padded_sequence(torch.cat((mean, maximum), -1), block_lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, final = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=blocks)
        valid = torch.arange(blocks, device=x.device)[None, :] < block_lengths[:, None]
        weights = torch.softmax(self.attention(output).squeeze(-1).masked_fill(~valid, -torch.inf), 1)
        memory = (weights[..., None] * output).sum(1)
        return self.head(torch.cat((final[-1], memory), -1))


def make_model(architecture, dimension, classes=2):
    if architecture == 'gru': return SequenceClassifier(dimension, classes, 32)
    if architecture == 'pool': return PoolClassifier(dimension, classes)
    if architecture == 'hierarchical': return HierarchicalMemory(dimension, classes)
    raise ValueError('Unknown architecture.')


def probabilities(model, x, lengths, batch=256):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            logits = model(torch.as_tensor(x[start:start+batch]), torch.as_tensor(lengths[start:start+batch]))
            result.append(torch.softmax(logits, -1).numpy())
    return np.concatenate(result)


def train(architecture, training, validation, config, seed):
    torch.manual_seed(seed)
    x, lengths, y = training
    v, vl, vy = validation
    model = make_model(architecture, x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    data = [torch.tensor(a) for a in (x, lengths, y)]
    val = [torch.tensor(a) for a in (v, vl, vy)]
    best_loss, stale, best_epoch, history = float('inf'), 0, -1, []
    best = copy.deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        total = 0.
        for rows in torch.randperm(len(x), generator=generator).split(config.batch_size):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(data[0][rows], data[1][rows]), data[2][rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        model.eval()
        losses = []
        with torch.inference_mode():
            for start in range(0, len(v), config.batch_size):
                end = start + config.batch_size
                losses.append((float(nn.functional.cross_entropy(model(val[0][start:end], val[1][start:end]), val[2][start:end])), len(val[2][start:end])))
        loss = sum(a*b for a,b in losses)/len(v)
        history.append({'epoch': epoch, 'train_loss': total/len(x), 'val_loss': loss})
        if epoch % 10 == 0:
            print(f'seed {seed} {architecture} epoch {epoch+1} val_loss={loss:.5f}', flush=True)
        if loss < best_loss:
            best_loss, best_epoch, stale = loss, epoch, 0
            best = copy.deepcopy(model.state_dict())
        elif epoch + 1 >= config.min_epochs:
            stale += 1
            if stale >= config.patience: break
    model.load_state_dict(best)
    history.append({'best_epoch': best_epoch, 'best_val_loss': best_loss, 'training_seconds': time.perf_counter()-started})
    return model, history
