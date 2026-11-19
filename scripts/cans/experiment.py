"""CANS accuracy experiment on the frozen ScopeAudit tasks.

Rules: uniform, hard-negative, CANS-hard, CANS-weighted, CANS-uniform.
The pilot model is a fixed logistic regression; the final models are the same
fixed logistic/forest/MLP as the ScopeAudit experiment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/scope_audit'))
import experiment as sa  # noqa: E402
sa.AUDIT_REPLICATES = 5

OUT = ROOT / 'results/cans_v1'
JOBS = ROOT / 'results/scope_audit_v1/jobs'
ARRAYS = ROOT / 'results/scope_audit_v1/arrays'
CACHE = ROOT / 'data/processed/contrapositive_sampling_v1'
BASE = ROOT / 'results/scope_audit_v1'
FRACTIONS = [0.10, 0.30]
RULES = ['uniform', 'hard', 'cans_hard', 'cans_weighted', 'cans_uniform',
         'zeros_uniform', 'zeros_hard', 'cans_mix', 'cans_adaptive', 'risk_weighted']
MODELS = ['logistic', 'forest', 'mlp']
AUDIT_FRACTIONS = [0.10, 0.20]
EPSILON = 0.10


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_source(source):
    with np.load(CACHE / (source + '.npz')) as archive:
        return {key: archive[key] for key in archive.files}


def pilot_model(x, positive, negative, seed):
    index = np.r_[positive, negative]
    labels = np.r_[np.ones(len(positive), dtype=np.int8),
                   np.zeros(len(negative), dtype=np.int8)]
    model = LogisticRegression(C=1, max_iter=250, solver='lbfgs', random_state=seed)
    model.fit(x[index], labels)
    return model


def stratum_bounds(members, hidden_mask, audit_fraction, job_seed, target, hiding):
    """One-sided Clopper-Pearson upper bounds per stratum for one audit draw."""
    labels = sorted(members)
    sizes = np.array([len(members[j]) for j in labels])
    alloc = sa.allocation_for(sizes, audit_fraction)
    rng = np.random.default_rng(job_seed + target * 1009 + int(hiding * 100000)
                                + 77000 + [0.02, 0.05, 0.10, 0.20].index(audit_fraction) * 7919)
    m = max(1, len(labels))
    alpha = 0.05 / m
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


def fit_predict_weighted(x, positive, negative, y, test, seed, negative_mass,
                         model_name, negative_weights):
    index = np.r_[positive, negative]
    labels = np.r_[np.ones(len(positive), dtype=np.int8),
                   np.zeros(len(negative), dtype=np.int8)]
    weights = np.r_[np.ones(len(positive)),
                    negative_weights / negative_weights.sum() * negative_mass]
    if model_name == 'logistic':
        model = LogisticRegression(C=1, max_iter=250, random_state=seed, solver='lbfgs')
    elif model_name == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=48, max_depth=10, min_samples_leaf=5,
                                       n_jobs=1, random_state=seed)
    else:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(32,), learning_rate_init=.001, max_iter=60,
                              tol=1e-4, n_iter_no_change=10, random_state=seed,
                              early_stopping=False)
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(x[index], labels, sample_weight=weights)
    probability = model.predict_proba(x[test])[:, 1]
    return probability, sa.score_metrics(y[test], probability)


def select_rule(rule, candidate, pilot_score, x, seed, budget, members, hidden_mask,
                job_seed, target, hiding):
    rng = np.random.default_rng(seed)
    if rule == 'uniform':
        return np.sort(rng.choice(candidate, size=budget, replace=False)), dict(
            audit_cost=0, certified_n=0, certified_coverage=None, mean_bound=None,
            pooled_bound=None, audit_fraction=None, status='complete')
    if rule == 'hard':
        order = np.argsort(-pilot_score, kind='stable')[:budget]
        return np.sort(candidate[order]), dict(
            audit_cost=0, certified_n=0, certified_coverage=None, mean_bound=None,
            pooled_bound=None, audit_fraction=None, status='complete')
    # CANS: find the smallest audit fraction whose certified set can fill the budget.
    certified = None
    summary = None
    used_fraction = None
    for audit_fraction in AUDIT_FRACTIONS:
        audit_index = [0.02, 0.05, 0.10, 0.20].index(audit_fraction)
        summary_i, replicates = sa.run_certification(
            members, hidden_mask, audit_fraction, audit_index,
            job_seed + target * 1009 + int(hiding * 100000) + 77000)
        key = f'certified_indices__e{EPSILON:g}'
        certified_i = np.sort(replicates[0].get(key, np.array([], dtype=int)))
        if len(certified_i) >= budget:
            certified = certified_i
            summary = summary_i
            used_fraction = audit_fraction
            break
    if certified is None:
        return None, dict(status='infeasible', audit_cost=None, certified_n=None,
                          certified_coverage=None, mean_bound=None, pooled_bound=None,
                          audit_fraction=None)
    if rule == 'cans_uniform':
        selected = rng.choice(certified, size=budget, replace=False)
    else:
        cert_score = pilot_model(x, positive, [], seed).predict_proba(x[certified])[:, 1] \
            if False else None
        # Reuse the pilot scores from the caller by re-scoring the certified set.
        selected = None  # replaced below
    # The caller supplies a scorer for the certified set; this branch is handled there.
    return certified, dict(
        status='complete', audit_cost=float(summary['audit_cost']),
        certified_n=int(len(certified)),
        certified_coverage=float(len(certified) / len(candidate)),
        mean_bound=float(summary['mean_bound']),
        pooled_bound=float(summary['pooled_bound']),
        audit_fraction=float(used_fraction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--fractions', default='0.10,0.30')
    parser.add_argument('--rules', default=','.join(RULES))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--tag', default='main')
    args = parser.parse_args()
    fractions = [float(x) for x in args.fractions.split(',')]
    rules = args.rules.split(',')
    models = args.models.split(',')
    OUT.mkdir(parents=True, exist_ok=True)

    dispositions = pd.read_csv(BASE / 'job_dispositions.csv')
    dispositions = dispositions[(dispositions.status == 'complete')
                                & (dispositions.model_eligible)]
    jobs = dispositions.job.tolist()
    if args.limit:
        jobs = jobs[:args.limit]
    data_cache = {}
    rows = []
    for job_name in jobs:
        job = json.loads((JOBS / (job_name + '.json')).read_text())
        if job['source'] not in data_cache:
            data_cache[job['source']] = load_source(job['source'])
        data = data_cache[job['source']]
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
        hidden_mask = np.zeros(len(y), dtype=bool)
        hidden_mask[hidden] = True
        members = sa.stratum_members(candidate, strata)
        for fraction in fractions:
            budget = max(1, int(np.floor(len(candidate) * fraction)))
            pilot_negative = np.random.default_rng(
                job['seed'] + job['target'] * 1009 + int(job['hiding_fraction'] * 100000)
                + 31337 + int(fraction * 10000)).choice(candidate, size=budget, replace=False)
            pilot = pilot_model(x, positive, pilot_negative, job['seed'])
            pilot_score = pilot.predict_proba(x[candidate])[:, 1]
            for rule in rules:
                base = dict(job=job_name, source=job['source'], target=job['target'],
                            seed=job['seed'], hiding_fraction=job['hiding_fraction'],
                            fraction=fraction, budget=budget, rule=rule,
                            candidate_n=len(candidate))
                if rule in ('uniform', 'hard', 'zeros_uniform', 'zeros_hard'):
                    if rule == 'uniform':
                        selected = np.sort(np.random.default_rng(
                            job['seed'] + job['target'] * 1009 + int(fraction * 10000)
                            + 424242).choice(candidate, size=budget, replace=False))
                    elif rule == 'hard':
                        order = np.argsort(-pilot_score, kind='stable')[:budget]
                        selected = np.sort(candidate[order])
                    else:
                        pool = known_zero
                        if len(pool) < budget:
                            rows.append(dict(**base, model=None, status='infeasible',
                                             reason='known_zero_smaller_than_budget'))
                            continue
                        if rule == 'zeros_uniform':
                            selected = np.sort(np.random.default_rng(
                                job['seed'] + job['target'] * 1009 + int(fraction * 10000)
                                + 626262).choice(pool, size=budget, replace=False))
                        else:
                            pool_score = pilot.predict_proba(x[pool])[:, 1]
                            order = np.argsort(-pool_score, kind='stable')[:budget]
                            selected = np.sort(pool[order])
                    meta = dict(status='complete', audit_cost=0, certified_n=0,
                                certified_coverage=None, mean_bound=None,
                                pooled_bound=None, audit_fraction=None)
                elif rule == 'risk_weighted':
                    selected = np.sort(np.random.default_rng(
                        job['seed'] + job['target'] * 1009 + int(fraction * 10000)
                        + 424242).choice(candidate, size=budget, replace=False))
                    bounds, audit_cost = stratum_bounds(
                        members, hidden_mask, 0.10, job['seed'], job['target'],
                        job['hiding_fraction'])
                    selected_bounds = np.array([bounds.get(int(s), 1.0)
                                                for s in strata[selected]])
                    negative_weights = 1.0 - np.minimum(0.5, selected_bounds)
                    meta = dict(status='complete', audit_cost=float(audit_cost),
                                certified_n=0, certified_coverage=None,
                                mean_bound=float(np.mean(list(bounds.values()))),
                                pooled_bound=float(np.mean(list(bounds.values()))),
                                audit_fraction=0.10,
                                risk_weight_mean=float(negative_weights.mean()))
                else:
                    certified, meta = select_rule(
                        rule, candidate, pilot_score, x, job['seed'], budget, members,
                        hidden_mask, job['seed'], job['target'], job['hiding_fraction'])
                    if certified is None:
                        rows.append(dict(**base, model=None, status='infeasible',
                                         reason='certified_set_smaller_than_budget',
                                         **meta))
                        continue
                    if rule == 'cans_uniform':
                        selected = np.sort(np.random.default_rng(
                            job['seed'] + job['target'] * 1009 + int(fraction * 10000)
                            + 515151).choice(certified, size=budget, replace=False))
                    elif rule in ('cans_mix', 'cans_adaptive'):
                        rng = np.random.default_rng(
                            job['seed'] + job['target'] * 1009 + int(fraction * 10000)
                            + 717171)
                        s_value = float(meta['pooled_bound']) if meta.get('pooled_bound') is not None else 1.0
                        if rule == 'cans_adaptive' and s_value < 0.01:
                            cert_score = pilot.predict_proba(x[certified])[:, 1]
                            order = np.argsort(-cert_score, kind='stable')[:budget]
                            selected = np.sort(certified[order])
                        else:
                            n_zero = min(len(known_zero), budget // 2)
                            zero_sel = rng.choice(known_zero, size=n_zero, replace=False)
                            n_cert = budget - n_zero
                            if n_cert > len(certified):
                                n_cert = len(certified)
                            cert_sel = rng.choice(certified, size=n_cert, replace=False)
                            selected = np.sort(np.concatenate([zero_sel, cert_sel]))
                    else:
                        cert_score = pilot.predict_proba(x[certified])[:, 1]
                        order = np.argsort(-cert_score, kind='stable')[:budget]
                        selected = np.sort(certified[order])
                nh = int(np.isin(selected, hidden).sum())
                base.update(negative_n=len(selected), hidden_selected=nh,
                            b_plus=nh / len(selected), **meta)
                if rule == 'cans_weighted' and meta.get('pooled_bound') is not None:
                    s = min(0.5, max(0.0, float(meta['pooled_bound'])))
                else:
                    s = 0.0
                base['correction_s'] = s
                negative_mass = len(known_zero) * (1.0 - s)
                for model_name in models:
                    if rule == 'risk_weighted':
                        probability, record = fit_predict_weighted(
                            x, positive, selected, y, test, job['seed'],
                            negative_mass, model_name, negative_weights)
                        record['negative_weight_mass'] = float(negative_weights.sum())
                    else:
                        probability, record = sa.fit_predict(
                            x, positive, selected, y, test, job['seed'],
                            negative_mass, model_name)
                    rows.append(dict(**base, model=model_name,
                                     **{k: v for k, v in record.items()
                                        if k not in ('warnings', 'negative_n')}))
        print(job_name, 'complete', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / ('all_results_' + args.tag + '.csv'), index=False)
    print(frame.groupby(['source', 'fraction', 'rule', 'model']).size())


if __name__ == '__main__':
    main()
