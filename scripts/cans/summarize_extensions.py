"""Summarize the full distribution-family and RCSS runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/cans_v1'
SEED = 20261118
BOOTSTRAP = 4000


def ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    rng = np.random.default_rng(SEED)
    means = [np.mean(rng.choice(values, len(values), replace=True))
             for _ in range(BOOTSTRAP)]
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def paired(data, reference_rule, rules, metrics=('ap', 'top5_recall', 'brier')):
    rows = []
    for (source, model), block in data.groupby(['source', 'model']):
        reference = block[block.rule == reference_rule].set_index('job')
        for rule in rules:
            current = block[block.rule == rule].set_index('job')
            common = reference.index.intersection(current.index)
            if not len(common):
                continue
            for metric in metrics:
                delta = (current.loc[common, metric].to_numpy(dtype=float)
                         - reference.loc[common, metric].to_numpy(dtype=float))
                low, high = ci(delta)
                rows.append(dict(source=source, model=model, reference=reference_rule,
                                 rule=rule, metric=metric, jobs=len(common),
                                 delta=float(delta.mean()), ci90_low=low, ci90_high=high,
                                 equivalent=bool(low is not None and low > -0.02 and high < 0.02),
                                 b_plus_delta=float(
                                     (current.loc[common, 'b_plus']
                                      - reference.loc[common, 'b_plus']).mean())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', default=str(OUT / 'distribution_results_distribution_all.csv'))
    parser.add_argument('--rcss', default=str(OUT / 'rcss_results_rcss_rice_full.csv'))
    args = parser.parse_args()
    dist = pd.read_csv(args.distribution)
    dist = dist[dist.status == 'complete']
    dist_rules = ['bernoulli', 'gaussian', 'poisson', 'gradient', 'hard']
    dist_summary = paired(dist, 'uniform', dist_rules)
    dist_summary.to_csv(OUT / 'distribution_equivalence_all.csv', index=False)
    print('DISTRIBUTION AP (rice)')
    print(dist_summary[(dist_summary.metric == 'ap')
                       & (dist_summary.source == 'rice')].to_string(index=False))
    print('\nDISTRIBUTION mean b_+ by rule (rice)')
    print(dist[dist.source == 'rice'].groupby('rule')['b_plus'].mean().to_string())

    rcss = pd.read_csv(args.rcss)
    rcss = rcss[rcss.status == 'complete']
    rcss_rules = ['hard', 'zeros_uniform', 'cans_hard', 'cans_mix', 'rcss_e10']
    rcss_summary = paired(rcss, 'uniform', rcss_rules)
    rcss_summary.to_csv(OUT / 'rcss_equivalence_full.csv', index=False)
    print('\nRCSS AP (rice)')
    print(rcss_summary[rcss_summary.metric == 'ap'].to_string(index=False))
    print('\nRCSS mean b_+ by rule')
    print(rcss.groupby('rule')['b_plus'].mean().to_string())


if __name__ == '__main__':
    main()
