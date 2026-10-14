"""Frozen protocol sona-eval-1.0.0: sizes, conditions, seeds and acceptance gates.

Changing anything here changes the protocol hash; runs refuse to resume across
a hash change. Amendments must be dated in docs/SONA_EVALUATION.md.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

PROTOCOL_VERSION = 'sona-eval-1.0.0'

DEV_SEED = 52001
BASELINE_SEEDS = (52011, 52037, 52069)
# Seeds already used by earlier studies; new runs must not reuse them.
USED_SEEDS = frozenset({7, 1007, 11, 29, 47, 71, 101, 451, 479, 503, 541, 577, 701, 733, 769,
                        811, 827, 853, 941, 967, 983, 1103, 1129, 1151})

CONDITIONS = ('familiar', 'slower', 'long_sparse', 'short', 'dense', 'sparse',
              'noisy_timing', 'lookalike_heavy', 'mixed')
CHAIN_SPLITS = ('chain_familiar', 'chain_long')
FAMILIAR_GAPS = (0, 2, 4, 8, 16, 32)
UNSEEN_GAPS = (64, 128, 256)
BUDGETS = (.01, .02, .05)
PRIMARY_BUDGET = .02
TRUNCATIONS = (8, 32)

ATTACK_KINDS = ('scan', 'bruteforce', 'beacon')
LOOKALIKE_KINDS = ('inventory', 'polling', 'retries', 'user_auth')

# Allowed adapter inference columns. Everything else is label, key or provenance.
INFERENCE_COLUMNS = ('flow_id', 'session_id', 'event_time', 'duration_s', 'src_port', 'dst_port',
                     'protocol', 'service', 'packets_sent', 'packets_recv', 'bytes_sent',
                     'bytes_recv', 'outcome', 'prior_gap_s')
TRAINING_LABEL_COLUMNS = ('window_target', 'attack_flow')


@dataclasses.dataclass(frozen=True)
class Sizes:
    train_windows: int = 2400        # generalization windows, half attack
    train_triples: int = 200         # chain triples (three windows each)
    val_windows: int = 2400
    val_triples: int = 200
    test_windows: int = 800          # per condition
    chain_familiar_triples: int = 180
    chain_long_triples: int = 180


SMOKE_SIZES = Sizes(train_windows=160, train_triples=24, val_windows=160, val_triples=24,
                    test_windows=48, chain_familiar_triples=18, chain_long_triples=12)


@dataclasses.dataclass(frozen=True)
class TrainRecipe:
    """Counterpart recipe; identical for every replicate."""
    epochs: int = 80
    min_epochs: int = 20
    patience: int = 12
    batch_size: int = 128
    flow_batch_size: int = 1024
    hidden_dim: int = 32
    threads: int = 2


SMOKE_RECIPE = TrainRecipe(epochs=3, min_epochs=1, patience=2)


@dataclasses.dataclass(frozen=True)
class Gates:
    g1_macro_lower_bound: float = 0.0
    g1_condition_floor: float = -.05
    g2_fpr_margin: float = .01
    g2_conditions: tuple[str, ...] = ('familiar', 'lookalike_heavy')
    g3_familiar_min: float = .80
    g3_unseen_triple_min: float = .70
    g3_shuffled_swap_max: float = .65
    g4_latency_ratio: float = 3.
    g4_memory_ratio: float = 3.
    g4_override_gain: float = .05
    g4_reference_batch: int = 32
    g4_reference_length: int = 256
    g4_reference_threads: int = 2
    min_final_replicates: int = 3
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20261013


def protocol_record(sizes: Sizes, recipe: TrainRecipe) -> dict:
    return {'protocol_version': PROTOCOL_VERSION, 'sizes': dataclasses.asdict(sizes),
            'recipe': dataclasses.asdict(recipe), 'gates': dataclasses.asdict(Gates()),
            'conditions': list(CONDITIONS), 'chain_splits': list(CHAIN_SPLITS),
            'familiar_gaps': list(FAMILIAR_GAPS), 'unseen_gaps': list(UNSEEN_GAPS),
            'budgets': list(BUDGETS), 'primary_budget': PRIMARY_BUDGET,
            'truncations': list(TRUNCATIONS), 'inference_columns': list(INFERENCE_COLUMNS)}


def record_hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


# Files that determine generated cases, scores, thresholds and audits. Reporting
# and efficiency scripts post-process saved outputs and are hashed separately.
RUN_SOURCES = ('__init__.py', 'protocol.py', 'cases.py', 'metrics.py', 'adapters.py', 'audit.py', 'run.py')


def suite_source_hashes() -> dict[str, str]:
    here = Path(__file__).parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in RUN_SOURCES}
