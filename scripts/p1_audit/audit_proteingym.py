"""P1 ProteinGym negative-label audit.

Zero-shot tail audit: for each assay and model, select the bottom q fraction of
model scores as negative labels and count measured high-fitness variants in that
tail.  Official leaderboard rank comes from the ProteinGym summary CSV.
"""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/p1/proteingym'
DMS_DIR = ROOT / 'data/processed/p1_audit/proteingym/dms/DMS_ProteinGym_substitutions'
OUT = ROOT / 'results/p1_audit_v1'
SCORES_ZIP = RAW / 'zero_shot_substitutions_scores.zip'
DMS_ZIP = RAW / 'DMS_ProteinGym_substitutions_v1.3.zip'
SUMMARY = RAW / 'Summary_performance_DMS_substitutions_Spearman.csv'
DMS_LEVEL = RAW / 'DMS_substitutions_Spearman_DMS_level.csv'
META = RAW / 'DMS_substitutions.csv'

QUANTILES = [0.05, 0.10, 0.20]
HIGH_QUANTILE = 0.75
LOW_QUANTILE = 0.25
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261118


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n')


def load_dms():
    data = {}
    for path in sorted(DMS_DIR.glob('*.csv')):
        frame = pd.read_csv(path, usecols=['mutant', 'DMS_score'])
        frame = frame.dropna(subset=['mutant', 'DMS_score'])
        frame = frame.drop_duplicates('mutant')
        data[path.stem] = frame
    return data


def inspect_scores():
    zf = zipfile.ZipFile(SCORES_ZIP)
    members = [m for m in zf.namelist() if m.lower().endswith('.csv')]
    print('members', len(members))
    for member in members[:10]:
        with zf.open(member) as stream:
            frame = pd.read_csv(stream, nrows=3)
        print(member, frame.shape, frame.columns.tolist()[:8])
    return members


def assay_frames(zf, members):
    """Yield (assay, frame, model_columns) reading only useful columns."""
    non_model = {'mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin'}
    for member in members:
        with zf.open(member) as stream:
            header = pd.read_csv(stream, nrows=0)
        model_cols = [c for c in header.columns if c not in non_model]
        if not model_cols:
            continue
        usecols = ['mutant', 'DMS_score'] + model_cols
        with zf.open(member) as stream:
            frame = pd.read_csv(stream, usecols=usecols)
        yield Path(member).stem, frame, model_cols


def tail_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 20 or np.allclose(p, p[0]):
        return None
    high = y >= np.quantile(y, HIGH_QUANTILE)
    low = y <= np.quantile(y, LOW_QUANTILE)
    order = np.argsort(p, kind='stable')
    rows = []
    for q in QUANTILES:
        k = max(1, int(round(q * len(y))))
        selected = np.zeros(len(y), dtype=bool)
        selected[order[:k]] = True
        h = int(np.sum(selected & high))
        l = int(np.sum(selected & low))
        rows.append(dict(
            quantile=q, selected=int(k), high_n=int(high.sum()), low_n=int(low.sum()),
            hidden_selected=h,
            e_H=h / high.sum() if high.sum() else np.nan,
            b_plus=h / k,
            c_judged=h / (h + l) if (h + l) else np.nan,
            coverage=q))
    return rows


def main():
    global HIGH_QUANTILE
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--high-quantile', type=float, default=HIGH_QUANTILE)
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    if args.inspect:
        inspect_scores()
        return
    HIGH_QUANTILE = args.high_quantile
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dms = load_dms()
    summary = pd.read_csv(SUMMARY)
    dms_level = pd.read_csv(DMS_LEVEL).set_index('DMS ID')
    meta = pd.read_csv(META).set_index('DMS_id')
    zf = zipfile.ZipFile(SCORES_ZIP)
    members = [m for m in zf.namelist() if m.lower().endswith('.csv')]

    rows = []
    matched_assays = set()
    model_names = set()
    for assay, frame, model_cols in assay_frames(zf, members):
        fitness = frame['DMS_score'].to_numpy(dtype=float)
        for model in model_cols:
            score = frame[model].to_numpy(dtype=float)
            valid = np.isfinite(score) & np.isfinite(fitness)
            if valid.sum() < 20:
                continue
            metrics = tail_metrics(fitness[valid], score[valid])
            if metrics is None:
                continue
            rho = spearmanr(score[valid], fitness[valid]).correlation
            matched_assays.add(assay)
            model_names.add(model)
            for metric in metrics:
                rows.append(dict(assay=assay, model=model, spearman=rho,
                                 n=int(valid.sum()), **metric))

    audit = pd.DataFrame(rows)
    audit.to_csv(out / 'assay_audit.csv', index=False)

    # Merge metadata and official rank.
    meta_cols = ['selection_type', 'coarse_selection_type', 'MSA_Neff_L_category',
                 'DMS_total_number_mutants', 'source_organism', 'taxon']
    meta_small = meta[[c for c in meta_cols if c in meta.columns]].reset_index()
    meta_small = meta_small.rename(columns={'DMS_id': 'assay'})
    audit = audit.merge(meta_small, on='assay', how='left')
    audit.to_csv(out / 'assay_audit_with_metadata.csv', index=False)

    summary_small = summary[['Model_rank', 'Model_name', 'Average_Spearman',
                             'Model type']].rename(columns={'Model_name': 'model'})
    model_summary = (audit.groupby(['model', 'quantile'], as_index=False)
                     .agg(mean_e_H=('e_H', 'mean'), mean_b_plus=('b_plus', 'mean'),
                          mean_c_judged=('c_judged', 'mean'),
                          median_b_plus=('b_plus', 'median'),
                          assays=('assay', 'nunique'),
                          mean_spearman=('spearman', 'mean')))
    model_summary = model_summary.merge(summary_small, on='model', how='left')
    # The score columns use underscores; the official summary uses spaces and
    # parentheses.  Match on a normalized alphanumeric key.
    import re
    aliases = {
        'prosst20': 'prosstk20', 'prosst128': 'prosstk128',
        'prosst512': 'prosstk512', 'prosst1024': 'prosstk1024',
        'prosst2048': 'prosstk2048', 'prosst4096': 'prosstk4096',
        'esm3': 'esm3open14b', 'poet': 'poet200m',
        'saprot650maf2': 'saprot650m', 'saprot35maf2': 'saprot35m',
        'mulan_small': 'mulan', 'mulan': 'mulan',
    }
    def norm(value):
        key = re.sub(r'[^a-z0-9]', '', str(value).lower())
        return aliases.get(key, key)
    lookup = {norm(row['model']): (row['Model_rank'], row['Average_Spearman'],
                                   row['Model type'], row['model'])
              for row in summary_small.to_dict('records')}
    model_summary['official_name'] = model_summary['model']
    for index, row in model_summary.iterrows():
        key = norm(row['model'])
        if key in lookup:
            rank, avg, mtype, name = lookup[key]
            model_summary.loc[index, ['Model_rank', 'Average_Spearman', 'Model type', 'official_name']] = \
                [rank, avg, mtype, name]
    model_summary.to_csv(out / 'model_summary.csv', index=False)

    # Primary correlation: official average Spearman vs mean b_plus at q=0.10.
    report = dict(
        assays_in_scores=len(members), assays_matched=len(matched_assays),
        models_in_scores=len(model_names),
        models_with_official_rank=int(model_summary['Model_rank'].notna().sum()),
        quantiles=QUANTILES, high_quantile=HIGH_QUANTILE,
        low_quantile=LOW_QUANTILE,
        data=dict(scores_sha256=sha(SCORES_ZIP), dms_sha256=sha(DMS_ZIP),
                  summary_sha256=sha(SUMMARY), dms_level_sha256=sha(DMS_LEVEL),
                  metadata_sha256=sha(META)))
    for q in QUANTILES:
        block = model_summary[(model_summary['quantile'] == q)
                              & model_summary['Model_rank'].notna()]
        rho_b = spearmanr(block['Average_Spearman'], block['mean_b_plus'])
        rho_e = spearmanr(block['Average_Spearman'], block['mean_e_H'])
        top10 = block[block['Model_rank'] <= 10]
        rest = block[block['Model_rank'] > 10]
        if len(top10) and len(rest):
            u, p_value = mannwhitneyu(top10['mean_b_plus'], rest['mean_b_plus'],
                                      alternative='less')
        else:
            u, p_value = np.nan, np.nan
        report[f'q{q:g}'] = dict(
            models=len(block),
            spearman_official_vs_b_plus=float(rho_b.correlation),
            spearman_official_vs_b_plus_p=float(rho_b.pvalue),
            spearman_official_vs_e_H=float(rho_e.correlation),
            spearman_official_vs_e_H_p=float(rho_e.pvalue),
            top10_mean_b_plus=float(top10['mean_b_plus'].mean()) if len(top10) else None,
            rest_mean_b_plus=float(rest['mean_b_plus'].mean()) if len(rest) else None,
            mannwhitney_u=float(u) if np.isfinite(u) else None,
            mannwhitney_p=float(p_value) if np.isfinite(p_value) else None,
            top10_lower=bool(p_value < 0.05) if np.isfinite(p_value) else None)
    save_json(out / 'analysis_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
