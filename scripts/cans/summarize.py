"""Summarize CANS accuracy results with paired job bootstrap intervals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/cans_v1'
BOOTSTRAP = 2000
SEED = 20261118


def ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    rng = np.random.default_rng(SEED)
    means = [np.mean(rng.choice(values, len(values), replace=True))
             for _ in range(BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(OUT / 'all_results_adaptive_all.csv'))
    parser.add_argument('--tag', default='adaptive_all')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = data[data.status == 'complete'].copy()
    rows = []
    for (source, fraction, model), block in data.groupby(['source', 'fraction', 'model']):
        reference = block[block.rule == 'uniform'].set_index('job')
        for rule in ['zeros_uniform', 'cans_adaptive']:
            current = block[block.rule == rule].set_index('job')
            common = reference.index.intersection(current.index)
            if not len(common):
                continue
            ref = reference.loc[common]
            cur = current.loc[common]
            for metric in ['ap', 'top5_recall', 'brier']:
                delta = cur[metric].to_numpy(dtype=float) - ref[metric].to_numpy(dtype=float)
                low, high = ci(delta)
                rows.append(dict(source=source, fraction=fraction, model=model,
                                 contrast=f'{rule}_vs_uniform', metric=metric,
                                 jobs=len(common), reference_value=float(ref[metric].mean()),
                                 rule_value=float(cur[metric].mean()),
                                 delta=float(delta.mean()), ci_low=low, ci_high=high,
                                 positive=int((delta > 0).sum()),
                                 negative=int((delta < 0).sum())))
    # CANS versus the clean-zero baseline on the jobs where both are feasible.
    for (source, fraction, model), block in data.groupby(['source', 'fraction', 'model']):
        reference = block[block.rule == 'zeros_uniform'].set_index('job')
        current = block[block.rule == 'cans_adaptive'].set_index('job')
        common = reference.index.intersection(current.index)
        if not len(common):
            continue
        ref = reference.loc[common]
        cur = current.loc[common]
        for metric in ['ap', 'top5_recall', 'brier']:
            delta = cur[metric].to_numpy(dtype=float) - ref[metric].to_numpy(dtype=float)
            low, high = ci(delta)
            rows.append(dict(source=source, fraction=fraction, model=model,
                             contrast='cans_adaptive_vs_zeros_uniform', metric=metric,
                             jobs=len(common), reference_value=float(ref[metric].mean()),
                             rule_value=float(cur[metric].mean()),
                             delta=float(delta.mean()), ci_low=low, ci_high=high,
                             positive=int((delta > 0).sum()),
                             negative=int((delta < 0).sum())))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f'summary_{args.tag}.csv', index=False)
    print(result[result.metric == 'ap'].to_string(index=False))


if __name__ == '__main__':
    main()
