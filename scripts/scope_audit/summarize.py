"""Aggregate ScopeAudit results, risk--coverage frontiers and paired contrasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ROOT, OUT, BASE_RESULTS, SEEDS, HIDING_FRACTIONS, BUDGETS,
                    POLICIES, AUDIT_FRACTIONS, EPSILONS, MODELS,
                    PREDICTION_BUDGETS, save_csv, save_json, now)

BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261118


def load_jobs():
    registration = json.loads((OUT / 'registration.json').read_text())
    rows = []
    for source, target, seed, hiding in registration['jobs']:
        name = f'{source}__target{target:03d}__seed{seed}__h{int(round(hiding * 100)):02d}'
        path = OUT / 'jobs' / (name + '.json')
        if not path.exists():
            raise ValueError(f'Registered job missing: {name}')
        rows.append(json.loads(path.read_text()))
    return rows


def flatten(jobs, key):
    rows = []
    for job in jobs:
        for row in job.get(key, []):
            rows.append(row)
    return pd.DataFrame(rows)


def percentile(values, low=2.5, high=97.5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def bootstrap_ratio(numerator, denominator, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(denominator)
    if n == 0:
        return None, None
    estimates = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, n, size=n)
        d = denominator[index].sum()
        estimates.append(numerator[index].sum() / d if d else np.nan)
    return percentile(estimates)


def paired_bootstrap(numerators, denominator, reference, seed=BOOTSTRAP_SEED):
    """Bootstrap the difference of two ratios on the same resampled jobs."""
    rng = np.random.default_rng(seed)
    n = len(denominator)
    if n == 0:
        return None, None
    estimates = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, n, size=n)
        d = denominator[index].sum()
        if not d:
            estimates.append(np.nan)
            continue
        estimates.append((numerators[index].sum() - reference[index].sum()) / d)
    return percentile(estimates)


def main():
    jobs = load_jobs()
    audit = flatten(jobs, 'negative_audit')
    certification = flatten(jobs, 'certification')
    prediction = flatten(jobs, 'prediction')
    audit.to_csv(OUT / 'all_sampling_audits.csv', index=False)
    certification.to_csv(OUT / 'all_certification.csv', index=False)
    prediction.drop(columns=['warnings'], errors='ignore').to_csv(
        OUT / 'all_prediction_metrics.csv', index=False)
    dispositions = pd.DataFrame([
        dict(job=j['job'], source=j['source'], target=j['target'], seed=j['seed'],
             hiding_fraction=j['hiding_fraction'], status=j['status'],
             reason=j.get('reason', ''), model_eligible=j.get('model_eligible', False))
        for j in jobs])
    dispositions.to_csv(OUT / 'job_dispositions.csv', index=False)

    sampling_rows = []
    paired_rows = []
    frontier_rows = []
    audit_complete = audit[audit.status == 'complete'].copy()
    for source in ['rice', 'alga']:
        for hiding in HIDING_FRACTIONS:
            for fraction in BUDGETS:
                part = audit_complete[(audit_complete.source == source)
                                      & (audit_complete.hiding_fraction == hiding)
                                      & (audit_complete.fraction == fraction)]
                common_sets = {}
                for policy in ['uniform', 'context_deferred', 'matched_control']:
                    common_sets[policy] = set(part[part.policy == policy].job)
                common = set.intersection(*common_sets.values()) if all(common_sets.values()) else set()
                base = dict(source=source, hiding_fraction=hiding, fraction=fraction)
                if not common:
                    sampling_rows.append(dict(**base, policy='shared_comparison',
                                              status='no_common_feasible_jobs', jobs=0))
                    continue
                for policy in ['uniform', 'context_deferred', 'matched_control']:
                    values = part[(part.policy == policy) & part.job.isin(common)].sort_values('job')
                    numerator_e = values.hidden_selected.to_numpy(dtype=float)
                    denominator_e = values.hidden_n.to_numpy(dtype=float)
                    numerator_b = values.hidden_selected.to_numpy(dtype=float)
                    denominator_b = values.budget.to_numpy(dtype=float)
                    numerator_c = values.hidden_selected.to_numpy(dtype=float)
                    denominator_c = (values.hidden_selected + values.known_zero_selected).to_numpy(dtype=float)
                    ci_e = bootstrap_ratio(numerator_e, denominator_e)
                    ci_b = bootstrap_ratio(numerator_b, denominator_b)
                    sampling_rows.append(dict(
                        **base, policy=policy, status='complete', jobs=len(common),
                        targets=int(values.target.nunique()),
                        hidden_denominator=int(values.hidden_n.sum()),
                        hidden_selected=int(values.hidden_selected.sum()),
                        budget_total=int(values.budget.sum()),
                        e_H=float(values.hidden_selected.sum() / values.hidden_n.sum()),
                        e_H_ci_low=ci_e[0], e_H_ci_high=ci_e[1],
                        b_plus=float(values.hidden_selected.sum() / values.budget.sum()),
                        b_plus_ci_low=ci_b[0], b_plus_ci_high=ci_b[1],
                        c_judged=float(values.hidden_selected.sum()
                                       / max(1, (values.hidden_selected + values.known_zero_selected).sum())),
                        coverage=float(values.budget.sum() / values.candidate_n.sum()),
                        natural_unknown_fraction=float(values.natural_unknown_selected.sum()
                                                       / values.budget.sum()),
                        mean_eligible_fraction=float((values.eligible_n / values.candidate_n).mean()),
                        unavailable_jobs=int((part.policy == policy).sum() - len(common))))
                uniform = part[(part.policy == 'uniform') & part.job.isin(common)].sort_values('job')
                denominator_e = uniform.hidden_n.to_numpy(dtype=float)
                denominator_b = uniform.budget.to_numpy(dtype=float)
                for policy in ['context_deferred', 'matched_control']:
                    values = part[(part.policy == policy) & part.job.isin(common)].sort_values('job')
                    assert values.job.tolist() == uniform.job.tolist()
                    for metric, denominator in [('e_H', denominator_e), ('b_plus', denominator_b)]:
                        numerator = values.hidden_selected.to_numpy(dtype=float)
                        reference = uniform.hidden_selected.to_numpy(dtype=float)
                        delta = float((numerator.sum() - reference.sum()) / denominator.sum())
                        low, high = paired_bootstrap(numerator, denominator, reference)
                        paired_rows.append(dict(
                            **base, policy=policy, reference='uniform', metric=metric,
                            jobs=len(common), delta=delta, ci_low=low, ci_high=high))
                # Frontier points use the full complete set for each policy, not
                # only the paired common set, so feasibility is visible.
                for policy in ['uniform', 'context_deferred', 'matched_control']:
                    values = part[(part.policy == policy) & part.job.isin(common)]
                    if not len(values):
                        continue
                    frontier_rows.append(dict(
                        **base, kind='uncertified', policy=policy, jobs=len(values),
                        b_plus=float(values.hidden_selected.sum() / values.budget.sum()),
                        coverage=float(values.budget.sum() / values.candidate_n.sum()),
                        e_H=float(values.hidden_selected.sum() / values.hidden_n.sum()),
                        feasible_jobs=len(part[part.policy == policy])))
                # Audited-uniform: check the selected set and remove hidden positives.
                for policy in ['uniform', 'context_deferred']:
                    values = part[(part.policy == policy) & part.job.isin(common)].copy()
                    if not len(values):
                        continue
                    checked = values.budget - values.hidden_selected
                    frontier_rows.append(dict(
                        **base, kind='checked', policy='audited_' + policy, jobs=len(values),
                        b_plus=0.0, coverage=float(checked.sum() / values.candidate_n.sum()),
                        e_H=float(values.hidden_selected.sum() / values.hidden_n.sum()),
                        feasible_jobs=len(part[part.policy == policy])))

    save_csv(OUT / 'sampling_summary.csv', sampling_rows)
    save_csv(OUT / 'paired_contrasts.csv', paired_rows)

    # Certification summaries.  NaN b_plus means no stratum was certified;
    # report coverage over all jobs and b_plus over certifying jobs.
    candidate_lookup = (audit.drop_duplicates('job').set_index('job')['candidate_n'].to_dict())
    cert_rows = []
    for source in ['rice', 'alga']:
        for hiding in HIDING_FRACTIONS:
            part = certification[(certification.source == source)
                                 & (certification.hiding_fraction == hiding)]
            for audit_fraction in AUDIT_FRACTIONS:
                block = part[part.audit_fraction == audit_fraction]
                for epsilon in EPSILONS:
                    coverage_key = f'certified_coverage__e{epsilon:g}'
                    b_key = f'certified_b_plus__e{epsilon:g}'
                    certified = block[block[coverage_key] > 0]
                    pooled_accept_key = f'pooled_accept_fraction__e{epsilon:g}'
                    pooled_cover_key = f'pooled_coverage__e{epsilon:g}'
                    pooled_b_key = f'pooled_b_plus__e{epsilon:g}'
                    pooled_accepted = block[block[pooled_accept_key] > 0]
                    cert_rows.append(dict(
                        source=source, hiding_fraction=hiding,
                        audit_fraction=audit_fraction, epsilon=epsilon,
                        jobs=len(block),
                        certifying_jobs=len(certified),
                        certifying_fraction=len(certified) / len(block) if len(block) else None,
                        mean_coverage=float(block[coverage_key].mean()),
                        median_coverage=float(block[coverage_key].median()),
                        mean_coverage_when_certified=float(certified[coverage_key].mean()) if len(certified) else None,
                        mean_b_plus_when_certified=float(certified[b_key].mean()) if len(certified) else None,
                        max_b_plus_when_certified=float(certified[b_key].max()) if len(certified) else None,
                        mean_accepted_strata=float(block[f'accepted_strata__e{epsilon:g}'].mean()),
                        mean_calibration=float(block.calibration.mean()),
                        min_calibration=float(block.calibration.min()),
                        mean_audit_cost=float(block.audit_cost.mean()),
                        pooled_accept_fraction=float(block[pooled_accept_key].mean()),
                        pooled_mean_coverage=float(block[pooled_cover_key].mean()),
                        pooled_mean_b_plus_when_certified=(
                            float(pooled_accepted[pooled_b_key].mean()) if len(pooled_accepted) else None),
                        mean_audit_fraction=float(np.mean(
                            block.audit_cost.to_numpy()
                            / block.job.map(candidate_lookup).to_numpy(dtype=float)))))
    save_csv(OUT / 'certification_summary.csv', cert_rows)

    # Calibration across replicate audit samples.
    calibration_rows = []
    for source in ['rice', 'alga']:
        for hiding in HIDING_FRACTIONS:
            for audit_fraction in AUDIT_FRACTIONS:
                values = []
                for job in jobs:
                    if job.get('status') != 'complete':
                        continue
                    if job['source'] != source or job['hiding_fraction'] != hiding:
                        continue
                    for row in job.get('certification_replicates', []):
                        if row['audit_fraction'] == audit_fraction:
                            values.append(row['calibration'])
                if values:
                    values = np.asarray(values, dtype=float)
                    calibration_rows.append(dict(
                        source=source, hiding_fraction=hiding,
                        audit_fraction=audit_fraction, replicates=len(values),
                        mean_calibration=float(values.mean()),
                        min_calibration=float(values.min()),
                        fraction_at_least_095=float((values >= 0.95).mean())))
    save_csv(OUT / 'certification_calibration.csv', calibration_rows)

    # Pareto flags within each source/hiding over the uncertified frontier.
    frontier = pd.DataFrame(frontier_rows)
    if len(frontier):
        frontier['pareto'] = False
        for (source, hiding), block in frontier.groupby(['source', 'hiding_fraction']):
            block = block.reset_index()
            for i, row in block.iterrows():
                dominated = ((block.b_plus <= row.b_plus)
                             & (block.coverage >= row.coverage)
                             & ((block.b_plus < row.b_plus) | (block.coverage > row.coverage)))
                frontier.loc[block.loc[i, 'index'], 'pareto'] = not bool(dominated.any())
        frontier.to_csv(OUT / 'frontier.csv', index=False)

    # Prediction summaries: pair each policy with uniform on the same jobs.
    pred = prediction.copy()
    pred_rows = []

    def add_contrast(source, hiding, fraction, model, policy, current, reference,
                     policy_model=None, reference_model=None):
        current = current.set_index('job')
        common = reference.index.intersection(current.index)
        if not len(common):
            return
        joined = current.loc[common]
        base = reference.loc[common]
        for metric in ['ap', 'top5_recall', 'brier']:
            delta = joined[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
            low, high = percentile(delta)
            pred_rows.append(dict(
                source=source, hiding_fraction=hiding, fraction=fraction,
                model=model, policy_model=policy_model or model,
                reference_model=reference_model or model,
                policy=policy, jobs=len(common), metric=metric,
                policy_value=float(joined[metric].mean()),
                uniform_value=float(base[metric].mean()),
                delta=float(delta.mean()), ci_low=low, ci_high=high,
                delta_min=float(delta.min()), delta_max=float(delta.max())))

    for source in ['rice', 'alga']:
        for hiding in HIDING_FRACTIONS:
            for fraction in PREDICTION_BUDGETS:
                for model in MODELS:
                    block = pred[(pred.source == source) & (pred.hiding_fraction == hiding)
                                 & (pred.model == model)]
                    uniform = block[(block.policy == 'uniform') & (block.fraction == fraction)]
                    if not len(uniform):
                        continue
                    reference = uniform.set_index('job')
                    for policy in ['context_deferred', 'matched_control', 'audited_uniform']:
                        current = block[(block.policy == policy) & (block.fraction == fraction)]
                        add_contrast(source, hiding, fraction, model, policy, current, reference)
                    full = block[block.policy == 'full_observed']
                    add_contrast(source, hiding, fraction, model, 'full_observed', full, reference)
                    for policy in ['scope_audit', 'scope_audit_high']:
                        current = block[block.policy == policy]
                        add_contrast(source, hiding, fraction, model, policy, current, reference)
                    if model == 'logistic':
                        pu_block = pred[(pred.source == source) & (pred.hiding_fraction == hiding)
                                        & (pred.policy.isin(['upu', 'nnpu']))]
                        for policy in ['upu', 'nnpu']:
                            current = pu_block[pu_block.policy == policy]
                            add_contrast(source, hiding, fraction, model, policy, current, reference,
                                         policy_model='linear_pu', reference_model='logistic')
    save_csv(OUT / 'prediction_summary.csv', pred_rows)

    # Availability and registration facts.
    status_counts = dispositions.groupby(['source', 'status']).size().to_dict()
    report = dict(
        completed_utc=now(), registered_jobs=len(jobs),
        evaluable_jobs=int((dispositions.status == 'complete').sum()),
        not_evaluable_jobs=int((dispositions.status == 'not_evaluable').sum()),
        sampling_rows=len(audit), certification_rows=len(certification),
        prediction_rows=len(prediction), bootstrap_replicates=BOOTSTRAP,
        bootstrap_unit='common-feasible job, preserving target and split composition',
        status_counts={f'{k[0]}__{k[1]}': int(v) for k, v in status_counts.items()},
        old_results_used_for_new_estimands=False,
        natural_pool_purity_claimed=False,
        certification_assumption='random sampling within strata and exchangeability of checked and unchecked candidates')
    save_json(OUT / 'analysis_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
