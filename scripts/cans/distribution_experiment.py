"""Distribution-family negative samplers on the frozen rice tasks.

Reuses the old csane.filtering retention criteria: uniform, bernoulli,
gaussian, poisson, gradient, hard.  The pilot logistic supplies the logits;
the old calibration gives each criterion the same expected budget fraction.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/scope_audit'))
sys.path.insert(0, str(ROOT / 'src'))
import experiment as sa  # noqa: E402
from csane.filtering import retention_mask  # noqa: E402

OUT = ROOT / 'results/cans_v1'
JOBS = ROOT / 'results/scope_audit_v1/jobs'
ARRAYS = ROOT / 'results/scope_audit_v1/arrays'
CACHE = ROOT / 'data/processed/contrapositive_sampling_v1'
BASE = ROOT / 'results/scope_audit_v1'
FRACTION = 0.10
MODELS = ['logistic', 'forest', 'mlp']
CRITERIA = ['uniform', 'bernoulli', 'gaussian', 'poisson', 'gradient', 'hard']


def load_source(source):
    with np.load(CACHE / (source + '.npz')) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', default='rice')
    parser.add_argument('--max-per-source', type=int, default=None)
    parser.add_argument('--tag', default='distribution_rice')
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
        test = array['test']
        columns = array['feature_columns']
        observed = matrix[:, columns] >= 0
        raw_x = np.concatenate([(matrix[:, columns] == 1), observed], axis=1).astype(np.float32)
        scaler = StandardScaler().fit(raw_x[train])
        x = scaler.transform(raw_x)
        budget = max(1, int(np.floor(FRACTION * len(candidate))))
        rng = np.random.default_rng(job['seed'] + 31337)
        pilot_negative = rng.choice(candidate, size=budget, replace=False)
        pilot = LogisticRegression(C=1, max_iter=250, solver='lbfgs',
                                   random_state=job['seed'])
        pilot.fit(x[np.r_[positive, pilot_negative]],
                  np.r_[np.ones(len(positive)), np.zeros(len(pilot_negative))])
        probability = pilot.predict_proba(x[candidate])[:, 1]
        logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6)
                        / np.clip(1 - probability, 1e-6, 1))
        logits_t = torch.as_tensor(logits[:, None], dtype=torch.float32)
        labels_t = torch.zeros((len(candidate), 1), dtype=torch.float32)
        fraction = budget / len(candidate)
        for criterion in CRITERIA:
            generator = torch.Generator().manual_seed(job['seed'] + 9090
                                                      + CRITERIA.index(criterion))
            try:
                keep, inclusion = retention_mask(logits_t, labels_t, criterion,
                                                 fraction, generator)
                selected = candidate[keep[:, 0].numpy()]
            except Exception as exc:
                rows.append(dict(job=job_name, source=job['source'], rule=criterion,
                                 model=None, status='failed',
                                 error=type(exc).__name__ + ': ' + str(exc)[:200]))
                continue
            if len(selected) < 2:
                continue
            nh = int(np.isin(selected, hidden).sum())
            base = dict(job=job_name, source=job['source'], target=job['target'],
                        seed=job['seed'], hiding_fraction=job['hiding_fraction'],
                        rule=criterion, budget=budget, candidate_n=len(candidate),
                        negative_n=len(selected), hidden_selected=nh,
                        b_plus=nh / len(selected), audit_cost=0.0,
                        status='complete')
            for model_name in MODELS:
                probability_out, record = sa.fit_predict(
                    x, positive, selected, y, test, job['seed'],
                    len(known_zero), model_name)
                rows.append({**base, 'model': model_name,
                             **{k: v for k, v in record.items()
                                if k not in ('warnings', 'negative_n')}})
        print(job_name, 'done', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f'distribution_results_{args.tag}.csv', index=False)
    print(frame.groupby(['rule', 'model']).size())


if __name__ == '__main__':
    main()
