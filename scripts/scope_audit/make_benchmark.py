"""Package the ScopeAudit controlled-hiding benchmark."""
import json
from pathlib import Path

import pandas as pd

from common import ROOT, OUT, BASE_RESULTS, save_csv, save_json, sha, now

BENCH = ROOT / 'benchmark'


def main():
    BENCH.mkdir(exist_ok=True)
    dispositions = pd.read_csv(OUT / 'job_dispositions.csv')
    tasks = []
    for row in dispositions.itertuples():
        if row.status != 'complete':
            continue
        job = json.loads((OUT / 'jobs' / (row.job + '.json')).read_text())
        audit = job.get('negative_audit', [])
        if not audit:
            continue
        first = audit[0]
        tasks.append(dict(
            job=row.job, source=row.source, target=int(row.target), seed=int(row.seed),
            hiding_fraction=float(row.hiding_fraction),
            candidate_n=int(first['candidate_n']), hidden_n=int(first['hidden_n']),
            visible_positive_n=int(first['visible_positive_n']),
            known_zero_n=int(first['known_zero_n']),
            model_eligible=bool(row.model_eligible)))
    tasks = sorted(tasks, key=lambda x: (x['source'], x['target'], x['seed'], x['hiding_fraction']))
    save_json(BENCH / 'manifest.json', dict(
        name='ScopeAudit controlled-hiding benchmark',
        version='20261118',
        description='Negative-label candidate pools with known hidden positives under controlled hiding.',
        sources=['rice', 'alga'],
        seeds=sorted({t['seed'] for t in tasks}),
        hiding_fractions=sorted({t['hiding_fraction'] for t in tasks}),
        budgets=[0.05, 0.10, 0.20, 0.30, 0.50],
        tasks=tasks,
        metrics=['e_H', 'b_plus', 'c_judged', 'coverage', 'feasibility'],
        hidden_label_policy='hidden labels are used only by the evaluator, never by a rule',
        created_utc=now(),
        source_manifest_sha256=sha(BASE_RESULTS / 'data_manifest.json')))
    save_csv(BENCH / 'tasks.csv', tasks)

    # Data-use note: the benchmark ships indices and task definitions; the
    # source workbooks remain under their original terms.
    data_manifest = json.loads((BASE_RESULTS / 'data_manifest.json').read_text())
    lines = ['# Data use', '',
             'The benchmark ships task definitions and candidate indices, not the source workbooks.',
             'Obtain each source under its original terms and verify the SHA-256 values below.',
             '']
    for source in data_manifest['sources']:
        lines.append(f"- {source['name']}: `{source['source']}` "
                     f"(SHA-256 `{source['source_sha256']}`; cache `{source['cache_sha256']}`)")
    lines += ['', 'The frozen parsed cache used by this repository is at '
                  '`data/processed/contrapositive_sampling_v1/`.', '']
    (BENCH / 'DATA_USE.md').write_text('\n'.join(lines))

    leaderboard = []
    if (OUT / 'sampling_summary.csv').exists():
        summary = pd.read_csv(OUT / 'sampling_summary.csv')
        summary = summary[summary.status == 'complete']
        summary = summary[(summary.fraction.isin([0.10, 0.30]))
                          & (summary.policy.isin(['uniform', 'context_deferred', 'matched_control']))]
        for row in summary.itertuples():
            leaderboard.append(dict(
                source=row.source, hiding_fraction=row.hiding_fraction,
                fraction=row.fraction, policy=row.policy, jobs=row.jobs,
                e_H=row.e_H, b_plus=row.b_plus, c_judged=row.c_judged,
                coverage=row.coverage))
        save_csv(BENCH / 'leaderboard.csv', leaderboard)

    readme = f"""# ScopeAudit benchmark

Controlled-hiding tasks for auditing negative labels in plant functional data.
Each task is a `(source, target, seed, hiding_fraction)` tuple.  The evaluator
reconstructs the 70/30 entity split, hides a registered fraction of training
source positives, and scores a submitted candidate set.

## Files

- `manifest.json`: task list, seeds, hiding fractions, budgets and metrics.
- `tasks.csv`: one row per evaluable task.
- `evaluate.py`: reference evaluator.  Hidden labels are read only for scoring.
- `baselines.py`: reference implementations of uniform, context-deferred,
  matched-control and ScopeAudit-certified rules.
- `leaderboard.csv`: baseline summary at 10% and 30% budgets.
- `DATA_USE.md`: source access and hashes.

## Submission format

```json
{{
  "rice__target000__seed1907__h50": {{"selected": [12, 44, 91]}},
  "alga__target007__seed1907__h50": {{"certified": [3, 8, 13]}}
}}
```

Use `"selected"` for an uncertified negative sample and `"certified"` for a
set claimed under an audit.  Indices are row indices in the frozen parsed
matrix.  A submission is rejected if it contains a visible positive, a
non-candidate entity or a duplicate.

## Metrics

- `e_H`: selected hidden positives / all hidden positives (recall-like).
- `b_plus`: selected hidden positives / selected candidates (contamination-like).
- `c_judged`: selected hidden positives / (selected hidden positives + selected source zeros).
- `coverage`: selected candidates / candidate pool.
- Feasibility: whether the eligible pool can fill the registered budget.

`e_H` and `b_plus` answer different questions and are never substituted for
one another.  The percentages are conditional on the controlled hiding design;
they are not natural-pool contamination estimates.

## Reference

```bash
python baselines.py --rule uniform --fraction 0.10 --out uniform.json
python baselines.py --rule context_deferred --fraction 0.10 --out context.json
python baselines.py --rule scope_audit --audit-fraction 0.05 --epsilon 0.05 --out audit.json
python evaluate.py --submission audit.json --out audit_scores.json
```

The benchmark does not redistribute the source workbooks.  See `DATA_USE.md`.
"""
    (BENCH / 'README.md').write_text(readme)
    print(f'wrote {len(tasks)} tasks to {BENCH}')


if __name__ == '__main__':
    main()
