"""ScopeAudit controlled-hiding experiment (route A).

Registers a fresh set of jobs before running them, measures contamination and
coverage for four negative-sampling rules, runs stratified certification with a
Clopper--Pearson bound, and evaluates downstream models plus PU baselines.
"""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_name] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import warnings
import numpy as np
import sklearn
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

from common import (ROOT, OUT, CACHE, BASE_RESULTS, SEEDS, HIDING_FRACTIONS,
                    BUDGETS, POLICIES, AUDIT_FRACTIONS, AUDIT_REPLICATES,
                    EPSILONS, DELTA, MODELS, PREDICTION_BUDGETS,
                    save_json, sha, now, split, feature_columns)
from pu import estimate_positive_prior, fit_pu_logistic

DATA = {}


def initialize():
    threadpool_limits(1)
    for source in ['rice', 'alga']:
        with np.load(CACHE / (source + '.npz')) as archive:
            DATA[source] = {key: archive[key] for key in archive.files}
        DATA[source]['metadata'] = json.loads(
            (BASE_RESULTS / (source + '_target_dispositions.json')).read_text())


def candidate_pools(visible, train, coverage, other_positive, seed):
    """Return the four registered uncertified pools, with no hidden-label use."""
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
    assert len(matched) == len(deferred)
    for level in np.unique(strata[candidate]):
        assert (strata[matched] == level).sum() == (strata[deferred] == level).sum()
    pools = dict(uniform=candidate, context_deferred=deferred, matched_control=matched,
                 observed_reference=np.flatnonzero(train & (visible == 0)))
    return pools, edges


def scope_strata(candidate, coverage, visible, other_positive, edges):
    coverage_bin = np.searchsorted(edges, coverage, side='right')
    return 4 * coverage_bin + 2 * (visible == 0).astype(int) + other_positive.astype(int)


def allocation_for(sizes, audit_fraction):
    """Proportional allocation with at least one candidate per non-empty stratum."""
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


def stratum_members(candidate, strata):
    labels = np.unique(strata[candidate])
    return {int(j): candidate[strata[candidate] == j] for j in labels}


def run_certification(members, hidden_mask, audit_fraction, audit_index, seed):
    """Run repeated stratified certification and return summary rows."""
    sizes = np.array([len(members[j]) for j in sorted(members)])
    labels = np.array(sorted(members))
    alloc = allocation_for(sizes, audit_fraction)
    alloc_map = {int(j): int(n) for j, n in zip(labels, alloc)}
    true_positive = {int(j): int(hidden_mask[members[j]].sum()) for j in labels}
    true_rate = {int(j): true_positive[int(j)] / len(members[int(j)]) for j in labels}
    total_n = int(sum(len(members[int(j)]) for j in labels))
    total_hidden = int(sum(true_positive.values()))

    replicates = []
    for replicate in range(AUDIT_REPLICATES):
        rng = np.random.default_rng(seed + audit_index * 7919 + replicate * 104729)
        checked = {}
        successes = {}
        for j in labels:
            j = int(j)
            n = alloc_map[j]
            if n == 0:
                continue
            sample = rng.choice(members[j], size=n, replace=False)
            checked[j] = n
            successes[j] = int(hidden_mask[sample].sum())
        nonempty = sorted(checked)
        alpha_j = DELTA / max(1, len(nonempty))
        upper = {}
        for j in nonempty:
            n = checked[j]
            a = successes[j]
            upper[j] = 1.0 if a >= n else float(beta.ppf(1 - alpha_j, a + 1, n - a))
        calibration = float(np.mean([upper[j] >= true_rate[j] for j in nonempty])) if nonempty else np.nan
        pooled_n = int(sum(checked.values()))
        pooled_a = int(sum(successes.values()))
        pooled_bound = (1.0 if pooled_a >= pooled_n
                        else float(beta.ppf(1 - DELTA, pooled_a + 1, pooled_n - pooled_a)))
        row = dict(audit_fraction=audit_fraction, replicate=replicate,
                   audit_cost=int(sum(checked.values())),
                   nonempty_strata=len(nonempty),
                   calibration=calibration,
                   max_bound=max(upper.values()) if upper else np.nan,
                   mean_bound=float(np.mean(list(upper.values()))) if upper else np.nan,
                   pooled_n=pooled_n, pooled_a=pooled_a, pooled_bound=pooled_bound)
        for epsilon in EPSILONS:
            accepted = [j for j in nonempty if upper[j] <= epsilon]
            certified_n = int(sum(len(members[j]) for j in accepted))
            certified_hidden = int(sum(true_positive[j] for j in accepted))
            row[f'certified_n__e{epsilon:g}'] = certified_n
            row[f'certified_hidden__e{epsilon:g}'] = certified_hidden
            row[f'certified_b_plus__e{epsilon:g}'] = (
                certified_hidden / certified_n if certified_n else np.nan)
            row[f'certified_coverage__e{epsilon:g}'] = certified_n / sum(sizes)
            row[f'accepted_strata__e{epsilon:g}'] = len(accepted)
            row[f'certified_indices__e{epsilon:g}'] = (
                np.concatenate([members[j] for j in accepted]) if accepted else np.array([], dtype=int))
            pooled_accept = pooled_bound <= epsilon
            row[f'pooled_accept__e{epsilon:g}'] = int(pooled_accept)
            row[f'pooled_coverage__e{epsilon:g}'] = 1.0 if pooled_accept else 0.0
            row[f'pooled_b_plus__e{epsilon:g}'] = (
                total_hidden / total_n if pooled_accept and total_n else np.nan)
        replicates.append(row)

    summary = dict(audit_fraction=audit_fraction, audit_cost=float(np.mean([r['audit_cost'] for r in replicates])),
                   nonempty_strata=int(np.mean([r['nonempty_strata'] for r in replicates])),
                   calibration=float(np.nanmean([r['calibration'] for r in replicates])),
                   calibration_min=float(np.nanmin([r['calibration'] for r in replicates])),
                   calibration_max=float(np.nanmax([r['calibration'] for r in replicates])),
                   max_bound=float(np.nanmean([r['max_bound'] for r in replicates])),
                   mean_bound=float(np.nanmean([r['mean_bound'] for r in replicates])),
                   pooled_bound=float(np.nanmean([r['pooled_bound'] for r in replicates])))
    for epsilon in EPSILONS:
        key = f'__e{epsilon:g}'
        summary['certified_coverage' + key] = float(np.nanmean([r['certified_coverage' + key] for r in replicates]))
        summary['certified_b_plus' + key] = float(np.nanmean([r['certified_b_plus' + key] for r in replicates]))
        summary['certified_b_plus_max' + key] = float(np.nanmax([r['certified_b_plus' + key] for r in replicates]))
        summary['accepted_strata' + key] = float(np.nanmean([r['accepted_strata' + key] for r in replicates]))
        summary['pooled_coverage' + key] = float(np.mean([r['pooled_coverage' + key] for r in replicates]))
        summary['pooled_accept_fraction' + key] = float(np.mean([r['pooled_accept' + key] for r in replicates]))
        values = [r['pooled_b_plus' + key] for r in replicates
                  if np.isfinite(r['pooled_b_plus' + key])]
        summary['pooled_b_plus' + key] = float(np.mean(values)) if values else np.nan
    return summary, replicates


def score_metrics(y, p):
    order = np.argsort(-p, kind='stable')[:max(1, int(np.ceil(.05 * len(y))))]
    return dict(ap=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                top5_recall=float(y[order].sum() / y.sum()),
                test_n=len(y), test_positive=int(y.sum()))


def fit_predict(x, train_positive, negative, y, test, seed, negative_mass, model_name):
    index = np.r_[train_positive, negative]
    labels = np.r_[np.ones(len(train_positive), dtype=np.int8),
                   np.zeros(len(negative), dtype=np.int8)]
    weights = np.r_[np.ones(len(train_positive)),
                    np.full(len(negative), negative_mass / len(negative))]
    if model_name == 'logistic':
        model = LogisticRegression(C=1, max_iter=250, random_state=seed, solver='lbfgs')
    elif model_name == 'forest':
        model = RandomForestClassifier(n_estimators=48, max_depth=10, min_samples_leaf=5,
                                       n_jobs=1, random_state=seed)
    else:
        model = MLPClassifier(hidden_layer_sizes=(32,), learning_rate_init=.001, max_iter=60,
                              tol=1e-4, n_iter_no_change=10, random_state=seed,
                              early_stopping=False)
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(x[index], labels, sample_weight=weights)
    probability = model.predict_proba(x[test])[:, 1]
    assert np.isfinite(probability).all()
    return probability, dict(**score_metrics(y[test], probability),
                             seconds=time.monotonic() - start,
                             warnings=sorted({str(w.message) for w in captured}),
                             positive_n=len(train_positive), negative_n=len(negative),
                             negative_weight_mass=float(negative_mass))


def job_key(source, target, seed, hiding):
    return f'{source}__target{target:03d}__seed{seed}__h{int(round(hiding * 100)):02d}'


def run_job(job):
    source, target, seed, hiding = job
    name = job_key(*job)
    destination = OUT / 'jobs' / (name + '.json')
    artifact = OUT / 'arrays' / (name + '.npz')
    if destination.exists():
        cached = json.loads(destination.read_text())
        # A completed job is reusable only if its per-job array artifact is
        # present; otherwise the downstream CANS scripts cannot load it.
        if cached.get('status') != 'complete' or artifact.exists():
            return cached

    data = DATA[source]
    y = data['y'][:, target]
    train = split(data['groups'], seed)
    assert not set(data['groups'][train]) & set(data['groups'][~train])
    source_positive = np.flatnonzero(train & (y == 1))
    base = dict(job=name, source=source, target=target,
                target_name=data['metadata'][target]['target'], seed=seed,
                hiding_fraction=hiding)
    if len(source_positive) < 2:
        result = dict(**base, status='not_evaluable',
                      reason='fewer_than_two_training_positive_cells')
        save_json(destination, result)
        return result

    n_hidden = max(1, int(np.floor(hiding * len(source_positive))))
    hidden = np.sort(np.random.default_rng(seed + target * 1009 + int(hiding * 10000))
                     .choice(source_positive, size=n_hidden, replace=False))
    visible = y.copy()
    visible[hidden] = -1
    positive = np.flatnonzero(train & (visible == 1))
    known_zero = np.flatnonzero(train & (visible == 0))
    columns = feature_columns(source, target, data['metadata'])
    if not len(columns) or len(positive) < 2:
        result = dict(**base, status='not_evaluable',
                      reason='no_valid_input_context_or_visible_positive')
        save_json(destination, result)
        return result

    observed = data['y'][:, columns] >= 0
    other_positive = (data['y'][:, columns] == 1).any(axis=1)
    coverage = observed.sum(axis=1)
    pools, edges = candidate_pools(visible, train, coverage, other_positive,
                                   seed + target * 43 + int(hiding * 1000))
    candidate = pools['uniform']
    if not len(candidate):
        result = dict(**base, status='not_evaluable', reason='empty_candidate_pool')
        save_json(destination, result)
        return result
    hidden_mask = np.zeros(len(y), dtype=bool)
    hidden_mask[hidden] = True
    strata = scope_strata(candidate, coverage, visible, other_positive, edges)
    members = stratum_members(candidate, strata)

    raw_x = np.concatenate([(data['y'][:, columns] == 1), observed], axis=1).astype(np.float32)
    scaler = StandardScaler().fit(raw_x[train])
    x = scaler.transform(raw_x)

    test_index = np.flatnonzero(~train & (y >= 0))
    arrays = dict(hidden=hidden, train=np.flatnonzero(train),
                  test=test_index,
                  test_y=y[test_index],
                  candidate=candidate, coverage=coverage, strata=strata,
                  feature_columns=columns)

    audit_rows = []
    for fraction in BUDGETS:
        budget = max(1, int(np.floor(len(candidate) * fraction)))
        for p_index, policy in enumerate(POLICIES):
            pool = pools[policy]
            row = dict(**base, fraction=fraction, policy=policy, budget=budget,
                       candidate_n=len(candidate), eligible_n=len(pool),
                       hidden_n=len(hidden), visible_positive_n=len(positive),
                       known_zero_n=len(known_zero))
            if len(pool) < budget:
                audit_rows.append(dict(**row, status='infeasible',
                                       reason='eligible_pool_smaller_than_fixed_budget'))
                continue
            rng = np.random.default_rng(seed + target * 1009 + int(fraction * 10000)
                                        + p_index * 1777 + int(hiding * 100000))
            selected = np.sort(rng.choice(pool, size=budget, replace=False))
            assert train[selected].all() and not np.isin(selected, positive).any()
            assert len(selected) == len(set(selected)) == budget
            nh = int(hidden_mask[selected].sum())
            nz = int((y[selected] == 0).sum())
            nu = int((y[selected] < 0).sum())
            assert nh + nz + nu == budget
            if policy == 'uniform' and fraction in PREDICTION_BUDGETS:
                arrays[f'selected__{policy}__q{fraction:g}'] = selected
            if fraction in PREDICTION_BUDGETS:
                arrays[f'selected__{policy}__q{fraction:g}'] = selected
            audit_rows.append(dict(
                **row, status='complete', selected_n=budget,
                hidden_selected=nh, known_zero_selected=nz,
                natural_unknown_selected=nu,
                e_H=nh / len(hidden), b_plus=nh / budget,
                c_judged=nh / (nh + nz) if (nh + nz) else None,
                coverage=budget / len(candidate),
                natural_unknown_fraction=nu / budget))

    certification_rows = []
    certification_replicates = []
    for audit_index, audit_fraction in enumerate(AUDIT_FRACTIONS):
        summary, replicates = run_certification(
            members, hidden_mask, audit_fraction, audit_index,
            seed + target * 1009 + int(hiding * 100000) + 77000)
        summary.update(**base, candidate_n=len(candidate), hidden_n=len(hidden))
        certification_rows.append(summary)
        for replicate in replicates:
            certification_replicates.append(dict(**base, **{
                k: v for k, v in replicate.items() if k != 'certified_indices__e5'
                and not k.startswith('certified_indices')}))
        if audit_fraction in (0.05, 0.20):
            for epsilon in EPSILONS:
                key = f'certified_indices__e{epsilon:g}'
                if key in replicates[0]:
                    arrays[f'certified__a{audit_fraction:g}__e{epsilon:g}'] = np.sort(replicates[0][key])

    model_eligible = bool(data['metadata'][target]['model_eligible']
                          and len(positive) >= 5
                          and (y[test_index] == 1).sum() >= 5
                          and (y[test_index] == 0).sum() >= 5)

    prediction_rows = []
    if model_eligible:
        negative_mass = len(known_zero)
        for fraction in PREDICTION_BUDGETS:
            for policy in POLICIES:
                key = f'selected__{policy}__q{fraction:g}'
                if key not in arrays:
                    continue
                selected = arrays[key]
                for model_name in MODELS:
                    probability, record = fit_predict(
                        x, positive, selected, y, arrays['test'], seed,
                        negative_mass, model_name)
                    arrays[f'prediction__{policy}__q{fraction:g}__{model_name}'] = probability
                    prediction_rows.append(dict(**base, policy=policy, fraction=fraction,
                                                model=model_name, **record))
        # Audited uniform: remove checked hidden positives from the selected set.
        for fraction in PREDICTION_BUDGETS:
            key = f'selected__uniform__q{fraction:g}'
            selected = arrays[key]
            checked = np.setdiff1d(selected, hidden, assume_unique=False)
            arrays[f'audited_uniform__q{fraction:g}'] = checked
            if len(checked) >= 2:
                for model_name in MODELS:
                    probability, record = fit_predict(
                        x, positive, checked, y, arrays['test'], seed,
                        negative_mass, model_name)
                    arrays[f'prediction__audited_uniform__q{fraction:g}__{model_name}'] = probability
                    prediction_rows.append(dict(**base, policy='audited_uniform',
                                                fraction=fraction, model=model_name, **record))
        # ScopeAudit certified sets: the registered primary point and a
        # pre-declared high-assurance sensitivity (a = 0.20, epsilon = 0.10).
        for audit_fraction, epsilon, policy_name in [
                (0.05, 0.05, 'scope_audit'), (0.20, 0.10, 'scope_audit_high')]:
            key = f'certified__a{audit_fraction:g}__e{epsilon:g}'
            if key not in arrays or len(arrays[key]) < 2:
                continue
            certified = arrays[key]
            for model_name in MODELS:
                probability, record = fit_predict(
                    x, positive, certified, y, arrays['test'], seed,
                    negative_mass, model_name)
                arrays[f'prediction__{policy_name}__q{audit_fraction:g}__e{epsilon:g}__{model_name}'] = probability
                prediction_rows.append(dict(**base, policy=policy_name,
                                            fraction=audit_fraction,
                                            audit_fraction=audit_fraction,
                                            epsilon=epsilon,
                                            model=model_name, **record))
        # PU baselines on all unlabeled candidates.
        prior, prior_unlabeled = estimate_positive_prior(x, positive, candidate, seed)
        for pu_name, nnpu in [('upu', False), ('nnpu', True)]:
            probability = fit_pu_logistic(
                x, positive, candidate, x[arrays['test']], seed, prior, nnpu=nnpu)
            arrays[f'prediction__{pu_name}'] = probability
            prediction_rows.append(dict(**base, policy=pu_name, fraction=0.0,
                                        model='linear_pu', prior=prior,
                                        prior_unlabeled=prior_unlabeled,
                                        **score_metrics(y[arrays['test']], probability)))
        # Full-observed reference.
        for model_name in MODELS:
            probability, record = fit_predict(
                x, positive, known_zero, y, arrays['test'], seed,
                negative_mass, model_name)
            arrays[f'prediction__full_observed__{model_name}'] = probability
            prediction_rows.append(dict(**base, policy='full_observed', fraction=0.0,
                                        model=model_name, **record))

    artifact = OUT / 'arrays' / (name + '.npz')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(artifact)
    result = dict(**base, status='complete', negative_audit=audit_rows,
                  certification=certification_rows,
                  certification_replicates=certification_replicates,
                  prediction=prediction_rows,
                  model_eligible=model_eligible,
                  arrays_sha256=sha(artifact), completed_utc=now())
    save_json(destination, result)
    return result


def hashlib_sha_bytes(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def build_jobs():
    jobs = []
    for source in ['rice', 'alga']:
        metadata = json.loads((BASE_RESULTS / (source + '_target_dispositions.json')).read_text())
        for target in metadata:
            if not target['sampling_eligible']:
                continue
            for seed in SEEDS:
                for hiding in HIDING_FRACTIONS:
                    jobs.append((source, target['target_index'], seed, hiding))
    return jobs


def main():
    jobs = build_jobs()
    registration = dict(
        created_scope='retrospective public sources; protocol and rules frozen before new outcomes',
        jobs=[list(j) for j in jobs], seeds=SEEDS, hiding_fractions=HIDING_FRACTIONS,
        budgets=BUDGETS, policies=POLICIES, audit_fractions=AUDIT_FRACTIONS,
        audit_replicates=AUDIT_REPLICATES, epsilons=EPSILONS, delta=DELTA,
        models=MODELS, prediction_budgets=PREDICTION_BUDGETS,
        experiment_sha256=sha(__file__),
        common_sha256=sha(ROOT / 'scripts/scope_audit/common.py'),
        pu_sha256=sha(ROOT / 'scripts/scope_audit/pu.py'),
        protocol_sha256=sha(ROOT / 'docs/scope_audit_protocol_20261118.md'),
        versions=dict(python=os.sys.version, numpy=np.__version__, sklearn=sklearn.__version__))
    path = OUT / 'registration.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous['jobs'] == registration['jobs'], 'Registered job list changed'
        assert previous['budgets'] == registration['budgets'], 'Registered budgets changed'
        assert previous['protocol_sha256'] == registration['protocol_sha256'], 'Protocol changed'
    else:
        save_json(path, registration)

    (OUT / 'jobs').mkdir(exist_ok=True)
    finished = []
    started = time.monotonic()

    def progress(status):
        save_json(OUT / 'experiment_progress.json', dict(
            status=status, pid=os.getpid(), expected_jobs=len(jobs),
            finished_jobs=len(finished),
            evaluable_jobs=sum(j['status'] == 'complete' for j in finished),
            model_fits=sum(len(j.get('prediction', [])) for j in finished),
            updated_utc=now(), elapsed_seconds=time.monotonic() - started))

    progress('running')
    try:
        with ProcessPoolExecutor(max_workers=8, initializer=initialize) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                finished.append(row)
                progress('running')
                print(f"{len(finished)}/{len(jobs)} {row['job']} {row['status']}", flush=True)
        progress('complete_pending_analysis')
    except BaseException as exc:
        progress('failed')
        save_json(OUT / 'experiment_failure.json',
                  dict(type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
