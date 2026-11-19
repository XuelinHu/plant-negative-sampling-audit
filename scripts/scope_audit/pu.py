"""Positive-unlabeled learning baselines for the ScopeAudit study.

Only a linear logistic model is used so that the PU risk and the audit-based
weighted logistic share the same hypothesis class.  The class prior is
estimated with five-fold cross-fitted P-vs-U probabilities (Elkan--Noto).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def estimate_positive_prior(x, positive, unlabeled, seed):
    """Return (overall_prior, prior_in_unlabeled).

    The P-vs-U classifier is fitted out of fold; only held-out unlabeled rows
    contribute to the estimate.  No test label is used.
    """
    n_pos = len(positive)
    n_unl = len(unlabeled)
    if n_pos == 0 or n_unl == 0:
        return 0.0, 0.0
    X = np.vstack([x[positive], x[unlabeled]])
    y = np.r_[np.ones(n_pos, dtype=np.int8), np.zeros(n_unl, dtype=np.int8)]
    predictions = np.full(n_unl, np.nan, dtype=float)
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, hold_index in folds.split(X):
        model = LogisticRegression(C=1.0, max_iter=250, solver='lbfgs', random_state=seed)
        model.fit(X[fit_index], y[fit_index])
        held_unlabeled = hold_index[y[hold_index] == 0]
        if len(held_unlabeled):
            local = held_unlabeled - n_pos
            predictions[local] = model.predict_proba(X[held_unlabeled])[:, 1]
    if np.isnan(predictions).any():
        predictions = np.nan_to_num(predictions, nan=float(np.nanmean(predictions)))
    prior_unlabeled = float(np.clip(predictions.mean(), 0.0, 0.99))
    observed_positive_fraction = n_pos / (n_pos + n_unl)
    overall = (n_pos + n_unl * prior_unlabeled) / (n_pos + n_unl)
    overall = float(np.clip(overall, observed_positive_fraction, 0.99))
    return overall, prior_unlabeled


def fit_pu_logistic(x, positive, unlabeled, x_test, seed, prior, nnpu=True,
                    max_iter=400, l2=1e-3):
    """Fit uPU or nnPU linear logistic and return test probabilities.

    The implementation is deliberately small and deterministic.  The PU risk
    follows du Plessis et al. (2015):
        R_uPU = pi R_P^+ + R_U^- - pi R_P^-
        R_nnPU = pi R_P^+ + max(0, R_U^- - pi R_P^-)
    """
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    Xp = torch.tensor(x[positive], dtype=torch.float32)
    Xu = torch.tensor(x[unlabeled], dtype=torch.float32)
    Xt = torch.tensor(x_test, dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], 1)
    bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-7, tolerance_change=1e-10,
        history_size=20, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        logits_p = model(Xp).squeeze(1)
        logits_u = model(Xu).squeeze(1)
        loss_p = bce(logits_p, torch.ones_like(logits_p))
        loss_u = bce(logits_u, torch.zeros_like(logits_u))
        loss_p_negative = bce(logits_p, torch.zeros_like(logits_p))
        negative_risk = loss_u - prior * loss_p_negative
        if nnpu:
            negative_risk = torch.clamp(negative_risk, min=0.0)
        risk = prior * loss_p + negative_risk
        regularizer = sum((parameter ** 2).sum() for parameter in model.parameters())
        loss = risk + l2 * regularizer
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        probability = torch.sigmoid(model(Xt).squeeze(1)).numpy()
    return probability.astype(float)
