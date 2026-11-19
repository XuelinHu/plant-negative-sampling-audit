"""Scientific training controls accept no hidden ground truth."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from csane.eligibility_losses import asymmetric_loss, nnpu_objective


class EligibilityMLP(nn.Module):
    def __init__(self, features, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, outputs))

    def forward(self, x):
        return self.net(x)


def online_hard_selection(logits, candidates, coverage=.5):
    """OHEM recomputes current negative BCE hardness each optimization step."""
    if not 0 < coverage <= 1 or logits.shape != candidates.shape:
        raise ValueError("invalid online hard-selection request")
    selected = torch.zeros_like(candidates)
    hardness = F.softplus(logits.detach())
    for output in range(logits.shape[1]):
        rows = torch.where(candidates[:, output])[0]
        count = int(coverage * len(rows))
        if count < 1:
            raise ValueError("empty OHEM pool for output")
        order = torch.argsort(hardness[rows, output], descending=True, stable=True)
        selected[rows[order[:count]], output] = True
    return selected


def mass_preserving_bce(logits, positive, candidates, selected, negative_weight=1.):
    if torch.any(selected & (~candidates | positive)):
        raise ValueError("positive or inadmissible relationship selected negative")
    counts, retained = candidates.sum(dim=0), selected.sum(dim=0)
    if torch.any((counts > 0) & (retained == 0)):
        raise ValueError("nonempty output population lost all negative coverage")
    weights = counts / retained.clamp(min=1)
    positive_loss = F.softplus(-logits)[positive].sum()
    negative_loss = (F.softplus(logits) * selected * weights[None, :]).sum()
    return (positive_loss + negative_weight * negative_loss) / (positive.sum() + candidates.sum())


def prediction_metrics(truth, observed, probability):
    aps, aucs = [], []
    for output in range(truth.shape[1]):
        y, p = truth[:, output][observed[:, output]], probability[:, output][observed[:, output]]
        if len(np.unique(y)) == 2:
            aps.append(average_precision_score(y, p))
            aucs.append(roc_auc_score(y, p))
    y, p = truth[observed], probability[observed]
    ece = 0.
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if mask.any():
            ece += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return dict(macro_ap=float(np.mean(aps)) if aps else None,
                micro_ap=float(average_precision_score(y, p)),
                macro_auroc=float(np.mean(aucs)) if aucs else None,
                fixed_half_f1=float(f1_score(y, p >= .5, zero_division=0)),
                brier=float(np.mean((p - y) ** 2)), ece10=float(ece), valid_outputs=len(aps))


def train_control(view, train_x, val_x, val_truth, val_observed, method, seed, selections, prior, epochs=40, negative_weight=None):
    """No truth/hidden arguments: validation truth selects checkpoint legitimately.

    Entry-wise gradients are saved without knowing which entries were hidden;
    the caller may subsequently evaluate them against held training truth.
    """
    if negative_weight is not None and (negative_weight < 0 or method in {"asl", "nnpu"}):
        raise ValueError("explicit nonnegative weight is only registered for BCE controls")
    torch.manual_seed(seed)
    model = EligibilityMLP(train_x.shape[1], view.positive.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx, vx = torch.from_numpy(train_x), torch.from_numpy(val_x)
    positive = torch.from_numpy(view.positive)
    candidates = torch.from_numpy(view.candidates)
    population = torch.from_numpy(view.admissible)
    selection_tensors = [torch.from_numpy(s) for s in selections]
    positive_gradient = np.zeros_like(view.positive, dtype=np.float64)
    signed_gradient = np.zeros_like(positive_gradient)
    history, best, best_state, selection_history = [], -np.inf, None, []
    best_positive_gradient, best_signed_gradient = None, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(tx)
        selected = selection_tensors[int(epoch > 20)]
        if method == "ohem":
            selected = online_hard_selection(logits, candidates, .5)
        selection_history.append(selected.detach().numpy().copy())
        if method == "asl":
            objective = asymmetric_loss(logits, positive, population)
        elif method == "nnpu":
            terms = [nnpu_objective(logits[:, j], positive[:, j], population[:, j], float(prior[j]))[0] * population[:, j].sum()
                     for j in range(logits.shape[1])]
            objective = sum(terms) / population.sum()
        else:
            weight = negative_weight if negative_weight is not None else .5 if method == "downweight" else 1.
            objective = mass_preserving_bce(logits, positive, candidates, selected, weight)
        if not torch.isfinite(objective):
            raise ValueError(f"nonfinite objective at epoch{epoch}")
        gradient = torch.autograd.grad(objective, logits, retain_graph=True)[0].detach().numpy()
        signed_gradient += gradient
        positive_gradient += np.maximum(gradient, 0)
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = model(vx).sigmoid().numpy()
        metrics = prediction_metrics(val_truth, val_observed, probability)
        history.append(dict(epoch=epoch, loss=float(objective.detach()), **metrics))
        criterion = metrics["macro_ap"]
        if criterion is not None and criterion > best:
            best, best_state = criterion, {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_positive_gradient, best_signed_gradient = positive_gradient.copy(), signed_gradient.copy()
    if best_state is None:
        raise ValueError("validation has no evaluable output")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probability = model(vx).sigmoid().numpy()
        training_probability = model(tx).sigmoid().numpy()
    return dict(model=model, history=history, validation_probability=validation_probability,
                training_probability=training_probability, positive_logit_gradient=positive_gradient,
                signed_logit_gradient=signed_gradient,
                best_prefix_positive_logit_gradient=best_positive_gradient,
                best_prefix_signed_logit_gradient=best_signed_gradient,
                selection_history=np.stack(selection_history),
                best_epoch=max(history, key=lambda row: row["macro_ap"] if row["macro_ap"] is not None else -np.inf)["epoch"])
