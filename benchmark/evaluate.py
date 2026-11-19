#!/usr/bin/env python3
"""Reference evaluator for the ScopeAudit controlled-hiding benchmark.

A submission is a JSON object keyed by job name.  Each entry supplies either
"selected" (an uncertified negative sample) or "certified" (a claimed
certified set).  The evaluator reconstructs the task from the frozen source
cache and the registered hiding seed; hidden labels are used only for scoring.

Example:
    python evaluate.py --submission my_submission.json --out my_scores.json

The benchmark does not redistribute the source workbooks.  Obtain them under
their original terms and place them at the paths recorded in DATA_USE.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/processed/contrapositive_sampling_v1'
BASE_RESULTS = ROOT / 'results/scope_audit_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(groups, seed):
    return np.array([
        int(hashlib.sha256(f'{seed}/{value}'.encode()).hexdigest()[:12], 16) / 16 ** 12 < 0.7
        for value in groups
    ])


def feature_columns(source, target, metadata):
    current = metadata[target]
    return np.array([
        row['target_index'] for row in metadata
        if row['primary'] and row['target_index'] != target
        and (source != 'alga' or (
            row['control_id'] != current['control_id']
            and row['treatment_id'] != current['treatment_id']))
    ], dtype=int)


def candidate_pools(visible, train, coverage, other_positive, seed):
    candidate = np.flatnonzero(train & (visible != 1))
    deferred = candidate[(coverage[candidate] > 0) & ~other_positive[candidate]]
    edges = np.unique(np.quantile(coverage[candidate], [1 / 3, 2 / 3]))
    strata = 2 * np.searchsorted(edges, coverage, side='right') + (visible == 0)
    rng = np.random.default_rng(seed + 42017)
    matched = []
    for level in np.unique(strata[deferred]):
        number = int((strata[deferred] == level).sum())
        available = candidate[strata[candidate] == level]
        matched.extend(rng.choice(available, size=number, replace=False).tolist())
    matched = np.sort(np.asarray(matched, dtype=int))
    return dict(uniform=candidate, context_deferred=deferred,
                matched_control=matched,
                observed_reference=np.flatnonzero(train & (visible == 0))), edges


def reconstruct(source, target, seed, hiding_fraction):
    with np.load(CACHE / (source + '.npz')) as archive:
        matrix = archive['y']
        y = matrix[:, target]
        groups = archive['groups']
    metadata = json.loads((BASE_RESULTS / (source + '_target_dispositions.json')).read_text())
    train = split(groups, seed)
    source_positive = np.flatnonzero(train & (y == 1))
    if len(source_positive) < 2:
        return None
    n_hidden = max(1, int(np.floor(hiding_fraction * len(source_positive))))
    hidden = np.sort(np.random.default_rng(
        seed + target * 1009 + int(hiding_fraction * 10000)).choice(
        source_positive, size=n_hidden, replace=False))
    visible = y.copy()
    visible[hidden] = -1
    positive = np.flatnonzero(train & (visible == 1))
    if len(positive) < 2:
        return None
    columns = feature_columns(source, target, metadata)
    if not len(columns):
        return None
    observed = matrix[:, columns] >= 0
    other_positive = (matrix[:, columns] == 1).any(axis=1)
    coverage = observed.sum(axis=1)
    pools, edges = candidate_pools(visible, train, coverage, other_positive,
                                   seed + target * 43 + int(hiding_fraction * 1000))
    return dict(source=source, target=target, seed=seed, hiding_fraction=hiding_fraction,
                y=y, train=train, hidden=hidden, positive=positive,
                known_zero=np.flatnonzero(train & (visible == 0)),
                candidate=pools['uniform'], pools=pools, coverage=coverage,
                other_positive=other_positive, columns=columns, edges=edges,
                metadata=metadata)


def score(task, selected, certificate=False):
    selected = np.asarray(selected, dtype=int)
    if selected.ndim != 1:
        raise ValueError('selected must be a flat index array')
    if len(selected) != len(set(selected.tolist())):
        raise ValueError('selected contains duplicates')
    candidate = set(task['candidate'].tolist())
    if not set(selected.tolist()) <= candidate:
        raise ValueError('selected contains a non-candidate entity')
    if np.isin(selected, task['positive']).any():
        raise ValueError('selected contains a visible positive')
    hidden = np.isin(selected, task['hidden'])
    known_zero = (task['y'][selected] == 0)
    natural_unknown = (task['y'][selected] < 0)
    n = len(selected)
    hidden_n = len(task['hidden'])
    result = dict(
        job=f"{task['source']}__target{task['target']:03d}__seed{task['seed']}"
            f"__h{int(round(task['hiding_fraction'] * 100)):02d}",
        source=task['source'], target=task['target'], seed=task['seed'],
        hiding_fraction=task['hiding_fraction'], candidate_n=len(task['candidate']),
        hidden_n=hidden_n, selected_n=n,
        hidden_selected=int(hidden.sum()),
        known_zero_selected=int(known_zero.sum()),
        natural_unknown_selected=int(natural_unknown.sum()),
        e_H=float(hidden.sum() / hidden_n),
        b_plus=float(hidden.sum() / n) if n else None,
        c_judged=(float(hidden.sum() / (hidden.sum() + known_zero.sum()))
                  if (hidden.sum() + known_zero.sum()) else None),
        coverage=float(n / len(task['candidate'])),
        natural_unknown_fraction=float(natural_unknown.sum() / n) if n else None,
        certificate=bool(certificate))
    return result


def aggregate(rows, reference_rows=None):
    import collections
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row['source'], row['hiding_fraction'])].append(row)
    out = []
    for (source, hiding), values in sorted(grouped.items()):
        hidden = sum(v['hidden_selected'] for v in values)
        denominator = sum(v['hidden_n'] for v in values)
        budget = sum(v['selected_n'] for v in values)
        known = sum(v['known_zero_selected'] for v in values)
        out.append(dict(source=source, hiding_fraction=hiding,
                        jobs=len(values), selected=budget,
                        hidden_selected=hidden,
                        e_H=hidden / denominator if denominator else None,
                        b_plus=hidden / budget if budget else None,
                        c_judged=hidden / (hidden + known) if (hidden + known) else None))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    submission = json.loads(Path(args.submission).read_text())
    rows = []
    for name, entry in submission.items():
        source, target, seed, hiding = name.split('__')
        target = int(target.replace('target', ''))
        seed = int(seed.replace('seed', ''))
        hiding = int(hiding.replace('h', '')) / 100.0
        task = reconstruct(source, target, seed, hiding)
        if task is None:
            raise ValueError(f'Task not evaluable: {name}')
        if 'selected' in entry:
            rows.append(score(task, entry['selected'], certificate=False))
        elif 'certified' in entry:
            rows.append(score(task, entry['certified'], certificate=True))
        else:
            raise ValueError(f'{name} needs selected or certified')
    result = dict(jobs=len(rows), aggregate=aggregate(rows), rows=rows)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
