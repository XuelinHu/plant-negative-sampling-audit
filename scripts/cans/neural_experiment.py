"""Complex-neural-network supplement for CANS.

Adds DeepMLP, tabular Transformer, inductive GraphSAGE and TabM to the frozen
10% negative-budget comparison.  A group-disjoint 80/20 split of the training
entities is used for early stopping; the registered test entities are untouched.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/scope_audit'))
sys.path.insert(0, str(ROOT / 'src'))
import experiment as sa  # noqa: E402
from csane.eligibility_encoders import FeatureTransformer, MeanGraphClassifier, graph_edges  # noqa: E402
from tabm import TabM  # noqa: E402
sa.AUDIT_REPLICATES = 5

OUT = ROOT / 'results/cans_v1'
JOBS = ROOT / 'results/scope_audit_v1/jobs'
ARRAYS = ROOT / 'results/scope_audit_v1/arrays'
CACHE = ROOT / 'data/processed/contrapositive_sampling_v1'
BASE = ROOT / 'results/scope_audit_v1'
FRACTION = 0.10
RULES = ['uniform', 'zeros_uniform', 'cans_adaptive']
MODELS = ['deep_mlp', 'tabm', 'graphsage', 'transformer']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def group_split(groups, seed, fraction):
    import hashlib
    return np.array([
        int(hashlib.sha256(f'{seed}/{value}'.encode()).hexdigest()[:12], 16) / 16 ** 12 < fraction
        for value in groups])


def load_source(source):
    with np.load(CACHE / (source + '.npz')) as archive:
        return {key: archive[key] for key in archive.files}


class DeepMLP(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def forward(model_name, model, x, train_x=None, train_edge=None, query_edge=None):
    if model_name == 'graphsage':
        if query_edge is None:
            return model(x, train_edge).squeeze(1)
        return model.query(train_x, x, train_edge, query_edge).squeeze(1)
    value = model(x)
    if value.ndim == 3:
        value = value.mean(dim=1)
    if value.ndim == 1:
        return value
    return value.squeeze(1)


def build_model(model_name, features):
    if model_name == 'deep_mlp':
        return DeepMLP(features)
    if model_name == 'tabm':
        return TabM.make(n_num_features=features, d_out=1, n_blocks=2, d_block=128,
                         k=8, dropout=0.1)
    if model_name == 'graphsage':
        return MeanGraphClassifier(features, 1)
    if model_name == 'transformer':
        return FeatureTransformer(features, 1)
    raise ValueError(model_name)


def train_and_score(model_name, x_fit, y_fit, w_fit, x_val, y_val, x_test, y_test, seed,
                    train_edge=None, val_edge=None, test_edge=None, epochs=60):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(model_name, x_fit.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    xf = torch.as_tensor(x_fit, dtype=torch.float32, device=DEVICE)
    yf = torch.as_tensor(y_fit, dtype=torch.float32, device=DEVICE)
    wf = torch.as_tensor(w_fit, dtype=torch.float32, device=DEVICE)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=DEVICE)
    xt = torch.as_tensor(x_test, dtype=torch.float32, device=DEVICE)
    tr_edge = None if train_edge is None else torch.as_tensor(train_edge, dtype=torch.long, device=DEVICE)
    va_edge = None if val_edge is None else torch.as_tensor(val_edge, dtype=torch.long, device=DEVICE)
    te_edge = None if test_edge is None else torch.as_tensor(test_edge, dtype=torch.long, device=DEVICE)
    best_ap, best_state, patience = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if model_name == 'graphsage':
            logits = forward(model_name, model, xf, train_edge=tr_edge)
            loss = (bce(logits, yf) * wf).sum() / wf.sum()
            loss.backward()
            optimizer.step()
        else:
            permutation = torch.randperm(len(xf), device=DEVICE)
            for begin in range(0, len(xf), 256):
                index = permutation[begin:begin + 256]
                logits = forward(model_name, model, xf[index])
                loss = (bce(logits, yf[index]) * wf[index]).sum() / wf.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            val_logits = forward(model_name, model, xv, train_x=xf,
                                 train_edge=tr_edge, query_edge=va_edge)
            val_probability = torch.sigmoid(val_logits).cpu().numpy()
        ap = average_precision_score(y_val, val_probability)
        if ap > best_ap + 1e-6:
            best_ap, patience = ap, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                break
    if best_state is None:
        raise ValueError('no validation checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = forward(model_name, model, xt, train_x=xf,
                              train_edge=tr_edge, query_edge=te_edge)
        probability = torch.sigmoid(test_logits).cpu().numpy()
    order = np.argsort(-probability, kind='stable')[:max(1, int(np.ceil(0.05 * len(y_test))))]
    return dict(ap=float(average_precision_score(y_test, probability)),
                auc=float(roc_auc_score(y_test, probability)),
                brier=float(brier_score_loss(y_test, probability)),
                top5_recall=float(y_test[order].sum() / y_test.sum()),
                best_val_ap=float(best_ap), epochs=epoch)


def select_negative(rule, candidate, known_zero, pilot, x, strata, hidden_mask,
                    budget, seed, target, hiding):
    rng = np.random.default_rng(seed)
    if rule == 'uniform':
        return np.sort(rng.choice(candidate, size=budget, replace=False)), dict(
            audit_cost=0, certified_coverage=None, pooled_bound=None, status='complete')
    if rule == 'zeros_uniform':
        if len(known_zero) < budget:
            return None, dict(status='infeasible', audit_cost=None,
                              certified_coverage=None, pooled_bound=None)
        return np.sort(rng.choice(known_zero, size=budget, replace=False)), dict(
            audit_cost=0, certified_coverage=None, pooled_bound=None, status='complete')
    if rule != 'cans_adaptive':
        raise ValueError(rule)
    # CANS adaptive: audit the fit candidate pool.
    members = sa.stratum_members(candidate, strata)
    certified = None
    summary = None
    for audit_fraction in (0.10, 0.20):
        audit_index = [0.02, 0.05, 0.10, 0.20].index(audit_fraction)
        summary_i, replicates = sa.run_certification(
            members, hidden_mask, audit_fraction, audit_index,
            seed + target * 1009 + int(hiding * 100000) + 77000)
        key = f'certified_indices__e0.1'
        certified_i = np.sort(replicates[0].get(key, np.array([], dtype=int)))
        if len(certified_i) >= budget:
            certified, summary = certified_i, summary_i
            break
    if certified is None:
        return None, dict(status='infeasible', audit_cost=None,
                          certified_coverage=None, pooled_bound=None)
    s = float(summary['pooled_bound']) if np.isfinite(summary['pooled_bound']) else 1.0
    if s < 0.01:
        score = pilot.predict_proba(x[certified])[:, 1]
        selected = certified[np.argsort(-score, kind='stable')[:budget]]
    else:
        n_zero = min(len(known_zero), budget // 2)
        zero_sel = rng.choice(known_zero, size=n_zero, replace=False)
        n_cert = budget - n_zero
        cert_sel = rng.choice(certified, size=min(n_cert, len(certified)), replace=False)
        selected = np.concatenate([zero_sel, cert_sel])
    return np.sort(selected), dict(
        audit_cost=float(summary['audit_cost']),
        certified_coverage=float(len(certified) / len(candidate)),
        pooled_bound=float(summary['pooled_bound']), status='complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-source', type=int, default=8)
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--rules', default=','.join(RULES))
    parser.add_argument('--tag', default='neural_subset')
    parser.add_argument('--sources', default='rice,alga')
    args = parser.parse_args()
    models = args.models.split(',')
    rules = args.rules.split(',')
    dispositions = pd.read_csv(BASE / 'job_dispositions.csv')
    dispositions = dispositions[(dispositions.status == 'complete')
                                & (dispositions.model_eligible)]
    jobs = []
    for source in args.sources.split(','):
        jobs += dispositions[dispositions.source == source].job.tolist()[:args.max_per_source]
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
        fit = train & group_split(data['groups'], job['seed'] + 7777, 0.8)
        val = train & ~fit
        hidden = array['hidden']
        visible = y.copy()
        visible[hidden] = -1
        positive_fit = np.flatnonzero(fit & (visible == 1))
        positive_val = np.flatnonzero(val & (visible == 1))
        zero_val = np.flatnonzero(val & (visible == 0))
        candidate_fit = np.flatnonzero(fit & (visible != 1))
        known_zero_fit = np.flatnonzero(fit & (visible == 0))
        test = array['test']
        if (len(positive_fit) < 5 or len(positive_val) < 5 or len(zero_val) < 5
                or len(candidate_fit) < 50):
            continue
        columns = array['feature_columns']
        observed = matrix[:, columns] >= 0
        raw_x = np.concatenate([(matrix[:, columns] == 1), observed], axis=1).astype(np.float32)
        scaler = StandardScaler().fit(raw_x[fit])
        x = scaler.transform(raw_x).astype(np.float32)
        budget = max(1, int(np.floor(FRACTION * len(candidate_fit))))
        pilot_rng = np.random.default_rng(job['seed'] + 31337)
        pilot_negative = pilot_rng.choice(candidate_fit, size=budget, replace=False)
        pilot = LogisticRegression(C=1, max_iter=250, solver='lbfgs',
                                   random_state=job['seed'])
        pilot.fit(x[np.r_[positive_fit, pilot_negative]],
                  np.r_[np.ones(len(positive_fit)), np.zeros(len(pilot_negative))])
        hidden_mask = np.zeros(len(y), dtype=bool)
        hidden_mask[hidden] = True
        strata = array['strata']
        for rule in rules:
            selected, meta = select_negative(
                rule, candidate_fit, known_zero_fit, pilot, x, strata, hidden_mask,
                budget, job['seed'], job['target'], job['hiding_fraction'])
            if selected is None:
                continue
            nh = int(np.isin(selected, hidden).sum())
            base = dict(job=job_name, source=job['source'], target=job['target'],
                        seed=job['seed'], hiding_fraction=job['hiding_fraction'],
                        rule=rule, budget=budget, candidate_n=len(candidate_fit),
                        negative_n=len(selected), hidden_selected=nh,
                        b_plus=nh / len(selected), **meta)
            negative_mass = len(known_zero_fit)
            w_fit = np.r_[np.ones(len(positive_fit)),
                          np.full(len(selected), negative_mass / len(selected))]
            y_fit = np.r_[np.ones(len(positive_fit)), np.zeros(len(selected))]
            y_val = np.r_[np.ones(len(positive_val)), np.zeros(len(zero_val))]
            x_val = x[np.r_[positive_val, zero_val]]
            for model_name in models:
                x_fit = x[np.r_[positive_fit, selected]]
                train_edge = val_edge = test_edge = None
                if model_name == 'graphsage':
                    train_edge = graph_edges(x_fit, x_fit, k=8)[0]
                    val_edge = graph_edges(x_fit, x_val, k=8)[1]
                    test_edge = graph_edges(x_fit, x[test], k=8)[1]
                try:
                    metrics = train_and_score(
                        model_name, x_fit, y_fit, w_fit,
                        x_val, y_val, x[test], y[test], job['seed'],
                        train_edge=train_edge, val_edge=val_edge, test_edge=test_edge)
                    rows.append({**base, 'model': model_name, 'status': 'complete', **metrics})
                except Exception as exc:
                    rows.append({**base, 'model': model_name, 'status': 'failed',
                                 'error': type(exc).__name__ + ': ' + str(exc)[:200]})
        print(job_name, 'done', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f'neural_results_{args.tag}.csv', index=False)
    print(frame.groupby(['source', 'rule', 'model']).size())


if __name__ == '__main__':
    main()
