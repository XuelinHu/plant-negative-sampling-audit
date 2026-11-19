# ScopeAudit benchmark

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
{
  "rice__target000__seed1907__h50": {"selected": [12, 44, 91]},
  "alga__target007__seed1907__h50": {"certified": [3, 8, 13]}
}
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
