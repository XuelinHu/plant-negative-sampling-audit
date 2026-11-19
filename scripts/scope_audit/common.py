"""Shared configuration for the ScopeAudit study (route A)."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/scope_audit_v1'
CACHE = ROOT / 'data/processed/contrapositive_sampling_v1'
BASE_RESULTS = ROOT / 'results/scope_audit_v1'
PAPER = ROOT / 'paper/accuracy_not_enough'

SEEDS = [1907, 2909, 3911, 4919, 5923]
HIDING_FRACTIONS = [0.25, 0.50, 0.75]
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
POLICIES = ['uniform', 'context_deferred', 'matched_control', 'observed_reference']
AUDIT_FRACTIONS = [0.02, 0.05, 0.10, 0.20]
AUDIT_REPLICATES = 20
EPSILONS = [0.01, 0.05, 0.10]
DELTA = 0.05
MODELS = ['logistic', 'forest', 'mlp']
PREDICTION_BUDGETS = [0.10, 0.30]


def sha(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def sanitize(value):
    """Make a result JSON-safe: non-finite floats become null."""
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(sanitize(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def save_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def split(groups, seed):
    import numpy as np
    return np.array([
        int(hashlib.sha256(f'{seed}/{value}'.encode()).hexdigest()[:12], 16) / 16 ** 12 < 0.7
        for value in groups
    ])


def feature_columns(source, target, metadata):
    import numpy as np
    current = metadata[target]
    return np.array([
        row['target_index'] for row in metadata
        if row['primary'] and row['target_index'] != target
        and (source != 'alga' or (
            row['control_id'] != current['control_id']
            and row['treatment_id'] != current['treatment_id']))
    ], dtype=int)
