# Plant negative-sampling audit

Code, benchmark and aggregate results for auditing negative labels in plant
functional data.

The public release contains the reproducibility materials only. It does not
contain the original source workbooks.

## What is included

- `benchmark/` — 660 controlled-hiding tasks, candidate-pool definitions, reference
  evaluator, baseline rules, leaderboard and data-use notes.
- `scripts/` — the controlled-hiding audit, CANS/distribution/RCSS experiments, the
  ProteinGym external audit, and the source-table encoder.
- `src/csane/` — the small set of model and filtering modules imported by the scripts.
- `data/processed/contrapositive_sampling_v1/` — frozen parsed response matrices used by
  the evaluator and audit scripts.
- `results/scope_audit_v1/` — registered design, task dispositions, summary tables and a
  compressed `jobs.tar.gz` with the per-task audit artifacts.
- `results/cans_v1/` — matched-performance, deep-model, distribution-family and RCSS
  result tables and reports.
- `results/p1_audit_v1/` — ProteinGym summary tables, reports and compressed per-assay
  audit tables.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

# Inspect the benchmark
python benchmark/evaluate.py --help

# Restore the per-task audit artifacts
tar -xzf results/scope_audit_v1/jobs.tar.gz -C results/scope_audit_v1

# Rebuild the audit arrays if they are absent, then rerun CANS
python scripts/scope_audit/experiment.py
python scripts/cans/experiment.py --fractions 0.10 \
  --rules uniform,zeros_uniform,cans_adaptive --models logistic,forest,mlp \
  --tag adaptive_10
```

The `scope_audit` step writes per-job `results/scope_audit_v1/arrays/*.npz` files. A
completed job is reused only when its array artifact is present, so a clean run rebuilds
the arrays required by the CANS scripts.

## Data

The benchmark uses parsed matrices derived from the KitBase rice phenotype supplement and
the *Chlamydomonas reinhardtii* gene-function screen. The original workbooks are **not**
redistributed; `benchmark/DATA_USE.md` records the access paths and SHA-256 values. The
frozen parsed cache in `data/processed/contrapositive_sampling_v1/` is included so that the
reference evaluator can run without the workbooks. Source workbooks remain subject to
their providers' terms.

`scripts/plant_resampling/prepare.py` documents how the parsed matrices were encoded. It
expects the source workbooks under
`data/processed/plant_counterinference/source_tables_v1/`.

## License

The analysis code is released under the MIT License (see `LICENSE`). The derived benchmark
data and aggregate results are released under the Creative Commons Attribution 4.0
International license (CC BY 4.0). The original source workbooks are not covered by these
licenses and are not redistributed.

## Citation

If you use these materials, please cite the archived release:

- Archived release: <https://doi.org/10.5281/zenodo.24362468>
- Code repository: <https://github.com/XuelinHu/plant-negative-sampling-audit>
