#!/usr/bin/env python3
"""Reference rules for the ScopeAudit benchmark.

    python baselines.py --rule uniform --fraction 0.10 --out submission.json
    python baselines.py --rule context_deferred --fraction 0.10 --out ...
    python baselines.py --rule scope_audit --audit-fraction 0.05 --epsilon 0.05 --out ...

The rules use only observable candidate features.  Hidden labels are never
read by a rule; the reference implementation obtains them only through the
audit sample, as an auditor would.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import evaluate

DELTA = 0.05


def scope_strata(task):
    coverage = task['coverage']
    candidate = task['candidate']
    edges = task['edges']
    coverage_bin = np.searchsorted(edges, coverage, side='right')
    visible_zero = task['y'] == 0
    return 4 * coverage_bin + 2 * visible_zero.astype(int) + task['other_positive'].astype(int)


def allocation_for(sizes, audit_fraction):
    nonempty = np.flatnonzero(sizes)
    target = max(len(nonempty), int(round(audit_fraction * sizes.sum())))
    raw = target * sizes / sizes.sum()
    alloc = np.floor(raw).astype(int)
    alloc[nonempty] = np.maximum(1, alloc[nonempty])
    alloc = np.minimum(alloc, sizes)
    while alloc.sum() > target:
        candidates = np.flatnonzero(alloc > 1)
        if not len(candidates):
            break
        j = candidates[np.argmax(alloc[candidates] - raw[candidates])]
        alloc[j] -= 1
    while alloc.sum() < target:
        candidates = np.flatnonzero(alloc < sizes)
        if not len(candidates):
            break
        j = candidates[np.argmax(raw[candidates] - alloc[candidates])]
        alloc[j] += 1
    return alloc


def select(task, rule, fraction, audit_fraction, epsilon):
    source, target, seed, hiding = task['source'], task['target'], task['seed'], task['hiding_fraction']
    candidate = task['candidate']
    if rule in ('uniform', 'context_deferred', 'matched_control', 'observed_reference'):
        pool = task['pools'][rule]
        budget = max(1, int(np.floor(len(candidate) * fraction)))
        if len(pool) < budget:
            return None
        policy_index = ['uniform', 'context_deferred', 'matched_control', 'observed_reference'].index(rule)
        rng = np.random.default_rng(seed + target * 1009 + int(fraction * 10000)
                                    + policy_index * 1777 + int(hiding * 100000))
        return np.sort(rng.choice(pool, size=budget, replace=False))
    if rule != 'scope_audit':
        raise ValueError(rule)

    strata = scope_strata(task)
    labels = np.unique(strata[candidate])
    members = {int(j): candidate[strata[candidate] == j] for j in labels}
    sizes = np.array([len(members[int(j)]) for j in labels])
    alloc = allocation_for(sizes, audit_fraction)
    alloc_map = {int(j): int(n) for j, n in zip(labels, alloc)}
    rng = np.random.default_rng(seed + target * 1009 + int(hiding * 100000) + 77000)
    hidden = set(task['hidden'].tolist())
    upper = {}
    for j in labels:
        j = int(j)
        n = alloc_map[j]
        if n == 0:
            continue
        sample = rng.choice(members[j], size=n, replace=False)
        a = sum(int(value) in hidden for value in sample)
        alpha_j = DELTA / max(1, len(labels))
        upper[j] = 1.0 if a >= n else float(beta.ppf(1 - alpha_j, a + 1, n - a))
    accepted = [j for j, bound in upper.items() if bound <= epsilon]
    if not accepted:
        return np.array([], dtype=int)
    return np.sort(np.concatenate([members[j] for j in accepted]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule', required=True)
    parser.add_argument('--fraction', type=float, default=0.10)
    parser.add_argument('--audit-fraction', type=float, default=0.05)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    tasks = json.loads((Path(__file__).parent / 'manifest.json').read_text())['tasks']
    if args.limit:
        tasks = tasks[:args.limit]
    submission = {}
    for entry in tasks:
        task = evaluate.reconstruct(entry['source'], entry['target'], entry['seed'],
                                    entry['hiding_fraction'])
        if task is None:
            continue
        selected = select(task, args.rule, args.fraction,
                          args.audit_fraction, args.epsilon)
        if selected is None:
            continue
        key = (f"{entry['source']}__target{entry['target']:03d}__seed{entry['seed']}"
               f"__h{int(round(entry['hiding_fraction'] * 100)):02d}")
        submission[key] = dict(selected=selected.tolist())
    Path(args.out).write_text(json.dumps(submission) + '\n')
    print(f'wrote {len(submission)} selections to {args.out}')


if __name__ == '__main__':
    main()
