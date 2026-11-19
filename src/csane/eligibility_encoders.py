"""Genuine common-input encoder controls; selectors and hidden truth are external."""
import copy
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F

from csane.eligibility_training import EligibilityMLP, prediction_metrics


class FeatureTransformer(nn.Module):
    """Transformer over feature coordinates, not a pretrained sequence encoder."""
    def __init__(self, features, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, 16) * .02)
        self.bias = nn.Parameter(torch.zeros(features, 16))
        self.cls = nn.Parameter(torch.zeros(1, 1, 16))
        layer = nn.TransformerEncoderLayer(16, 2, 64, .1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(16, outputs)

    def forward(self, x):
        tokens = x[:, :, None] * self.weight[None] + self.bias[None]
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.output(self.encoder(tokens)[:, 0])


class MeanGraphClassifier(nn.Module):
    """Actual two-layer inductive GraphSAGE over a label-free feature kNN graph."""
    def __init__(self, features, outputs):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.first = SAGEConv(features, 64, aggr="mean")
        self.second = SAGEConv(64, outputs, aggr="mean")

    def forward(self, x, edge):
        h = F.dropout(F.gelu(self.first(x, edge)), .1, self.training)
        return self.second(h, edge)

    def query(self, train_x, query_x, train_edge, query_edge):
        # Inductive inference on a combined graph: training nodes keep their
        # edges; query receivers are offset into the appended query block.
        n_train = len(train_x)
        combined_x = torch.cat([train_x, query_x], dim=0)
        q = query_edge.clone()
        q[1] += n_train
        combined_edge = torch.cat([train_edge, q], dim=1)
        return self.forward(combined_x, combined_edge)[n_train:]


def graph_edges(train_x, query_x, k=10):
    """All senders are training nodes; no label/annotation-derived edge is used."""
    k = min(k, len(train_x) - 1)
    if k < 1:
        raise ValueError("insufficient graph fitting nodes")
    nearest = NearestNeighbors(n_neighbors=k + 1).fit(train_x)
    train_neighbors = nearest.kneighbors(train_x, return_distance=False)
    neighbors = np.asarray([row[row != i][:k] for i, row in enumerate(train_neighbors)])
    queries = nearest.kneighbors(query_x, n_neighbors=k, return_distance=False)
    training = np.vstack([neighbors.ravel(), np.repeat(np.arange(len(train_x)), k)])
    query = np.vstack([queries.ravel(), np.repeat(np.arange(len(query_x)), k)])
    return training.astype(np.int64), query.astype(np.int64)


def supervision_weights(positive, admissible, selected):
    candidates = admissible & ~positive
    if np.any(selected & (~candidates | positive)):
        raise ValueError("encoder negative selection crossed domain/positive boundary")
    retained = selected.sum(axis=0)
    if np.any(retained == 0):
        raise ValueError("empty retained output")
    weights = positive.astype(np.float32) + selected * (candidates.sum(axis=0) / retained)[None]
    np.testing.assert_allclose(weights.sum(axis=0), admissible.sum(axis=0), rtol=1e-6)
    return weights.astype(np.float32)


def fit_classical(family, train, validation, selected, seed):
    weights = supervision_weights(train["positive"], train["admissible"], selected)
    models, native_history, warning_rows = [], [], []
    start = time.perf_counter()
    for output in range(train["positive"].shape[1]):
        keep = weights[:, output] > 0
        y = train["positive"][keep, output].astype(int)
        if len(np.unique(y)) != 2:
            raise ValueError(f"output{output} lacks supervised contrast")
        if family == "lr":
            model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
        elif family == "rf":
            model = RandomForestClassifier(n_estimators=64, min_samples_leaf=2, max_features="sqrt", n_jobs=1, random_state=seed + output)
        elif family == "boosting":
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=.1,
                l2_regularization=1., min_samples_leaf=10, early_stopping=False, random_state=seed + output)
        else:
            raise ValueError(f"unknown classical family:{family}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train["x"][keep], y, sample_weight=weights[keep, output])
        warning_rows.extend(dict(output=output, category=type(w.message).__name__, message=str(w.message)) for w in caught)
        native_history.append(dict(output=output, fit_rows=int(keep.sum()), weight_mass=float(weights[:, output].sum()),
            iterations=np.asarray(getattr(model, "n_iter_", [])).tolist(),
            trees=len(model.estimators_) if family == "rf" else None))
        models.append(model)
    validation_probability = np.column_stack([m.predict_proba(validation["x"])[:, 1] for m in models]).astype(np.float32)
    training_probability = np.column_stack([m.predict_proba(train["x"])[:, 1] for m in models]).astype(np.float32)
    metrics = prediction_metrics(validation["truth"], validation["observed"], validation_probability)
    return dict(model=models, history=[dict(stage="native fixed-budget terminal fit", **metrics)],
        native_history=native_history, warnings=warning_rows, validation_probability=validation_probability,
        training_probability=training_probability, best_epoch=None, seconds=time.perf_counter() - start)


def build_neural(family, features, outputs):
    if family == "mlp":
        return EligibilityMLP(features, outputs)
    if family == "transformer":
        return FeatureTransformer(features, outputs)
    if family == "graph":
        return MeanGraphClassifier(features, outputs)
    if family == "tabm":
        from tabm import TabM
        return TabM.make(n_num_features=features, d_out=outputs, n_blocks=2, d_block=128, k=8, dropout=.1)
    raise ValueError(f"unknown neural family:{family}")


def fit_neural(family, train, validation, selected, seed, device="cuda", epochs=40, batch_size=256):
    """Full-population loss accumulated by microbatches; one update per epoch.

    Validation truth legitimately selects a checkpoint. No hidden/future labels
    are passed or loaded here. Every selector uses identical row/microbatch order.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    weights = supervision_weights(train["positive"], train["admissible"], selected)
    model = build_neural(family, train["x"].shape[1], train["positive"].shape[1]).to(device)
    x, vx = torch.as_tensor(train["x"], device=device), torch.as_tensor(validation["x"], device=device)
    y = torch.as_tensor(train["positive"], dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, device=device)
    denominator = int(train["admissible"].sum())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_edge, query_edge = None, None
    if family == "graph":
        edges = graph_edges(train["x"], validation["x"])
        train_edge, query_edge = [torch.as_tensor(a, device=device) for a in edges]

    def entry_loss(logits, target):
        if logits.ndim == 3:
            return F.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits), reduction="none").mean(dim=1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    def probabilities(query):
        if family == "graph":
            logits = model.query(x, query, train_edge, query_edge) if query is vx else model(x, train_edge)
            return logits.sigmoid().detach().cpu().numpy()
        parts = []
        for begin in range(0, len(query), batch_size):
            value = model(query[begin:begin + batch_size]).sigmoid()
            parts.append((value.mean(dim=1) if value.ndim == 3 else value).detach().cpu().numpy())
        return np.concatenate(parts)

    history, best, best_state, best_epoch = [], -np.inf, None, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_value = 0.
        if family == "graph":
            loss = (entry_loss(model(x, train_edge), y) * w).sum() / denominator
            loss.backward()
            loss_value = float(loss.detach())
        else:
            for begin in range(0, len(x), batch_size):
                end = begin + batch_size
                loss = (entry_loss(model(x[begin:end]), y[begin:end]) * w[begin:end]).sum() / denominator
                loss.backward()
                loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = probabilities(vx)
        metrics = prediction_metrics(validation["truth"], validation["observed"], probability)
        history.append(dict(epoch=epoch, loss=loss_value, **metrics))
        if metrics["macro_ap"] is not None and metrics["macro_ap"] > best:
            best, best_epoch = metrics["macro_ap"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is None:
        raise ValueError("no usable validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probability = probabilities(vx)
        training_probability = probabilities(x)
    return dict(model=model, state_dict=best_state, history=history, native_history=[], warnings=[],
        best_epoch=best_epoch, validation_probability=validation_probability, training_probability=training_probability,
        seconds=time.perf_counter() - start, parameters=sum(p.numel() for p in model.parameters()),
        graph_edges=None if train_edge is None else dict(train=train_edge.cpu().numpy(), query=query_edge.cpu().numpy()))
