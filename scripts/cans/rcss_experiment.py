"""Risk-Constrained Stratified Sampler (RCSS).

Solves, per job:
    max sum_j b_j * hardness_j
    s.t. sum_j b_j = B, sum_j b_j * U_j <= epsilon * B, 0 <= b_j <= N_j,
where U_j is the one-sided Clopper-Pearson contamination bound for stratum j.
The LP is tiny (at most 12 strata); the selected candidates are the hardest
b_j candidates inside each stratum.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/scope_audit'))
import experiment as sa  # noqa: E402

OUT = ROOT / 'results/cans_v1'
JOBS = ROOT / 'results/scope_audit_v1/jobs'
ARRAYS = ROOT / 'results/scope_audit_v1/arrays'
CACHE = ROOT / 'data/processed/contrapositive_sampling_v1'
BASE = ROOT / 'results/scope_audit_v1'
FRACTION = 0.10
MODELS = ['logistic', 'forest', 'mlp']
RULES = ['uniform', 'hard', 'zeros_uniform', 'cans_hard', 'cans_mix',
         'rcss_e05', 'rcss_e10']
sa.AUDIT_REPLICATES = 5


def group_split(groups, seed, fraction):
    import hashlib
    return np.array([
        int(hashlib.sha256(f'{seed}/{value}'.encode()).hexdigest()[:12], 16) / 16 ** 12 < fraction
        for value in groups])


def load_source(source):
    with np.load(CACHE / (source + '.npz')) as archive:
        return {key: archive[key] for key in archive.files}


def pilot_score(candidate, x, positive, budget, seed):
    rng = np.random.default_rng(seed)
    negative = rng.choice(candidate, size=budget, replace=False)
    model = LogisticRegression(C=1, max_iter=250, solver='lbfgs', random_state=seed)
    model.fit(x[np.r_[positive, negative]],
              np.r_[np.ones(len(positive)), np.zeros(len(negative))])
    return model.predict_proba(x[candidate])[:, 1], model


def bounds_by_stratum(members, hidden_mask, audit_fraction, job_seed, target, hiding):
    labels = sorted(members)
    sizes = np.array([len(members[j]) for j in labels])
    alloc = sa.allocation_for(sizes, audit_fraction)
    audit_index = [0.02, 0.05, 0.10, 0.20].index(audit_fraction)
    rng = np.random.default_rng(job_seed + target * 1009 + int(hiding * 100000)
                                + 77000 + audit_index * 7919)
    alpha = 0.05 / max(1, len(labels))
    from scipy.stats import beta
    bounds = {}
    for j, n in zip(labels, alloc):
        j = int(j)
        n = int(n)
        if n <= 0:
            continue
        sample = rng.choice(members[j], size=n, replace=False)
        a = int(hidden_mask[sample].sum())
        bounds[j] = 1.0 if a >= n else float(beta.ppf(1 - alpha, a + 1, n - a))
    return bounds, int(alloc.sum())


def rcss_select(candidate, members, hidden_mask, scores, budget, epsilon,
                job_seed, target, hiding, audit_fraction=0.10):
    bounds, audit_cost = bounds_by_stratum(members, hidden_mask, audit_fraction,
                                           job_seed, target, hiding)
    labels = sorted(members)
    N = np.array([len(members[j]) for j in labels], dtype=float)
    U = np.array([bounds.get(int(j), 1.0) for j in labels], dtype=float)
    # mean pilot hardness per stratum
    lookup = {int(value): scores[index] for index, value in enumerate(candidate)}
    H = np.array([np.mean([lookup[int(v)] for v in members[j]]) for j in labels])
    if not np.any(N):
        return None, None
    result = linprog(-H, A_ub=U.reshape(1, -1), b_ub=epsilon * budget,
                     A_eq=np.ones((1, len(labels))), b_eq=budget,
                     bounds=[(0.0, float(n)) for n in N], method='highs')
    if not result.success:
        return None, dict(status='infeasible', audit_cost=audit_cost,
                          risk_bound=None, selected_n=0)
    alloc = np.floor(result.x).astype(int)
    alloc = np.minimum(alloc, N.astype(int))
    remaining = budget - int(alloc.sum())
    while remaining > 0:
        slack = epsilon * budget - float((U * alloc).sum())
        choices = [j for j in range(len(labels))
                   if alloc[j] < N[j] and U[j] <= slack + 1e-12]
        if not choices:
            break
        j = max(choices, key=lambda index: H[index])
        alloc[j] += 1
        remaining -= 1
    selected = []
    for j, take in zip(labels, alloc):
        if take <= 0:
            continue
        members_j = members[j]
        values = np.array([lookup[int(v)] for v in members_j])
        order = np.argsort(-values, kind='stable')[:take]
        selected.extend(members_j[order].tolist())
    selected = np.sort(np.asarray(selected, dtype=int))
    risk_bound = float((U * alloc).sum() / len(selected)) if len(selected) else None
    return selected, dict(status='complete', audit_cost=audit_cost,
                          risk_bound=risk_bound, selected_n=len(selected),
                          lp_risk=float((U * result.x).sum() / budget))


def cans_select(candidate, members, hidden_mask, scores, budget, job_seed,
                target, hiding, audit_fraction=0.10, epsilon=0.10):
    audit_index = [0.02, 0.05, 0.10, 0.20].index(audit_fraction)
    summary, replicates = sa.run_certification(
        members, hidden_mask, audit_fraction, audit_index,
        job_seed + target * 1009 + int(hiding * 100000) + 77000)
    certified = np.sort(replicates[0].get(f'certified_indices__e{epsilon:g}',
                                          np.array([], dtype=int)))
    if len(certified) < budget:
        return None, dict(status='infeasible', audit_cost=float(summary['audit_cost']),
                          certified_n=len(certified), certified_coverage=len(certified) / len(candidate))
    return certified, dict(status='complete', audit_cost=float(summary['audit_cost']),
                           certified_n=len(certified),
                           certified_coverage=len(certified) / len(candidate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', default='rice')
    parser.add_argument('--max-per-source', type=int, default=None)
    parser.add_argument('--tag', default='rcss_rice')
    args = parser.parse_args()
    dispositions = pd.read_csv(BASE / 'job_dispositions.csv')
    dispositions = dispositions[(dispositions.status == 'complete')
                                & (dispositions.model_eligible)]
    jobs = []
    for source in args.sources.split(','):
        block = dispositions[dispositions.source == source].job.tolist()
        if args.max_per_source:
            block = block[:args.max_per_source]
        jobs += block
    cache = {}
    rows = []
    for job_name in jobs:
        job = json.loads((JOBS / (job_name + '.json')).read_text())
        if job['source'] not in cache:
            cache[job['source']] = load_source(job['source'])
        data = cache[job['source']]
        array = np.load(ARRAYS / (job_name + '.npz'))
        matrix = data['y']
        y = matrix[:, job['target']]
        train = sa.split(data['groups'], job['seed'])
        hidden = array['hidden']
        visible = y.copy()
        visible[hidden] = -1
        positive = np.flatnonzero(train & (visible == 1))
        known_zero = np.flatnonzero(train & (visible == 0))
        candidate = array['candidate']
        strata = array['strata']
        test = array['test']
        columns = array['feature_columns']
        observed = matrix[:, columns] >= 0
        raw_x = np.concatenate([(matrix[:, columns] == 1), observed], axis=1).astype(np.float32)
        scaler = StandardScaler().fit(raw_x[train])
        x = scaler.transform(raw_x)
        budget = max(1, int(np.floor(FRACTION * len(candidate))))
        scores, pilot = pilot_score(candidate, x, positive, budget,
                                    job['seed'] + 31337)
        hidden_mask = np.zeros(len(y), dtype=bool)
        hidden_mask[hidden] = True
        members = sa.stratum_members(candidate, strata)
        for rule in RULES:
            if rule == 'uniform':
                selected = np.sort(np.random.default_rng(
                    job['seed'] + job['target'] * 1009 + 424242)
                    .choice(candidate, size=budget, replace=False))
                meta = dict(status='complete', audit_cost=0)
            elif rule == 'hard':
                order = np.argsort(-scores, kind='stable')[:budget]
                selected = np.sort(candidate[order])
                meta = dict(status='complete', audit_cost=0)
            elif rule == 'zeros_uniform':
                if len(known_zero) < budget:
                    continue
                selected = np.sort(np.random.default_rng(
                    job['seed'] + job['target'] * 1009 + 626262)
                    .choice(known_zero, size=budget, replace=False))
                meta = dict(status='complete', audit_cost=0)
            elif rule in ('cans_hard', 'cans_mix'):
                certified, meta = cans_select(
                    candidate, members, hidden_mask, scores, budget,
                    job['seed'], job['target'], job['hiding_fraction'])
                if certified is None:
                    continue
                if rule == 'cans_hard':
                    order = np.argsort(-scores[np.isin(candidate, certified)],
                                       kind='stable')[:budget]
                    selected = np.sort(certified[order])
                else:
                    rng = np.random.default_rng(job['seed'] + 717171)
                    n_zero = min(len(known_zero), budget // 2)
                    zero_sel = rng.choice(known_zero, size=n_zero, replace=False)
                    n_cert = min(budget - n_zero, len(certified))
                    cert_sel = rng.choice(certified, size=n_cert, replace=False)
                    selected = np.sort(np.concatenate([zero_sel, cert_sel]))
            elif rule.startswith('rcss'):
                epsilon = 0.05 if rule.endswith('e05') else 0.10
                selected, meta = rcss_select(
                    candidate, members, hidden_mask, scores, budget, epsilon,
                    job['seed'], job['target'], job['hiding_fraction'])
                if selected is None:
                    continue
            else:
                raise ValueError(rule)
            nh = int(np.isin(selected, hidden).sum())
            base = dict(job=job_name, source=job['source'], target=job['target'],
                        seed=job['seed'], hiding_fraction=job['hiding_fraction'],
                        rule=rule, budget=budget, candidate_n=len(candidate),
                        negative_n=len(selected), hidden_selected=nh,
                        b_plus=nh / len(selected), **meta)
            for model_name in MODELS:
                probability, record = sa.fit_predict(
                    x, positive, selected, y, test, job['seed'],
                    len(known_zero), model_name)
                rows.append({**base, 'model': model_name,
                             **{k: v for k, v in record.items()
                                if k not in ('warnings', 'negative_n')}})
        print(job_name, 'done', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f'rcss_results_{args.tag}.csv', index=False)
    print(frame.groupby(['source', 'rule', 'model']).size())


if __name__ == '__main__':
    main()
