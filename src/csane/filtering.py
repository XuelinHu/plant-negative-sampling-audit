"""Full-pool label-pair filtering with exact inclusion probabilities.

Rows are objects and columns are outputs. Positive losses are never filtered.
Independent unequal-probability Bernoulli retention has expected class-wise
size rho*N. Uniform and hard retention use exact rounded class-wise counts.
"""
from __future__ import annotations

import torch
from torch import Tensor


CRITERIA = ("uniform", "bernoulli", "gaussian", "poisson", "gradient", "hard")
OBJECTIVES = ("drop", "mass", "ht", "full", "scaled_full")


def retention_fraction(epoch: int, target: float, schedule: str = "static") -> float:
    if epoch < 1 or not 0 < target <= 1:
        raise ValueError("positive epoch and target in (0, 1] required")
    if schedule == "static":
        return 1.0 if epoch <= 20 else target
    if schedule == "staircase":
        if target != .25:
            raise ValueError("staircase is the preregistered 1/.75/.5/.25 schedule")
        if epoch <= 100:
            return 1.0
        if epoch <= 200:
            return .75
        if epoch <= 300:
            return .5
        return .25
    raise ValueError(schedule)


def _ranks(values: Tensor, negative: Tensor, descending: bool = False) -> Tensor:
    masked = values.detach().masked_fill(~negative, float("-inf") if descending else float("inf"))
    order = masked.argsort(dim=0, descending=descending, stable=True)
    rank = torch.empty_like(order)
    position = torch.arange(len(values), device=values.device)[:, None].expand_as(order)
    rank.scatter_(0, order, position)
    return rank


def calibrate_inclusion(weights: Tensor, negative: Tensor, fraction: float,
                        exploration: float = .25) -> Tensor:
    """Find pi_i with mean pi=rho in every nonempty output comparator pool.

    The uniform floor is epsilon*rho; hence HT weights are bounded by
    1/(epsilon*rho), not by the earlier with-replacement bound of four.
    """
    if weights.shape != negative.shape or weights.ndim != 2:
        raise ValueError("matching two-dimensional weights/mask required")
    if not 0 < fraction <= 1 or not 0 < exploration <= 1:
        raise ValueError("fractions must be in (0, 1]")
    with torch.no_grad():
        if fraction == 1:
            return torch.ones_like(weights)
        w = weights.detach().float().clamp_min(1e-6).masked_fill(~negative, 0)
        counts = negative.sum(dim=0).clamp_min(1)
        mean = w.sum(dim=0) / counts
        low = torch.zeros_like(mean)
        high = fraction / mean.clamp_min(1e-12)
        for _ in range(24):
            mass = (w * high).clamp_max(1).sum(dim=0) / counts
            insufficient = (mass < fraction - 1e-7) & negative.any(dim=0)
            if not bool(insufficient.any()):
                break
            high = torch.where(insufficient, high * 2, high)
        else:
            raise RuntimeError("could not bracket inclusion calibration")
        for _ in range(24):
            middle = (low + high) / 2
            mass = (w * middle).clamp_max(1).sum(dim=0) / counts
            low = torch.where(mass < fraction, middle, low)
            high = torch.where(mass >= fraction, middle, high)
        allocation = (w * ((low + high) / 2)).clamp_max(1)
        probability = exploration * fraction + (1 - exploration) * allocation
        return probability.masked_fill(~negative, 1)


def retention_mask(logits: Tensor, labels: Tensor, criterion: str, fraction: float,
                   generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or logits.shape != labels.shape:
        raise ValueError("logits and labels must be matching matrices")
    if criterion not in CRITERIA or not 0 < fraction <= 1:
        raise ValueError("invalid retention criterion/fraction")
    positive = labels > 0
    negative = ~positive
    with torch.no_grad():
        if fraction == 1:
            return torch.ones_like(positive), torch.ones_like(logits)
        counts = negative.sum(dim=0)
        if criterion in {"uniform", "hard"}:
            # A continuous uniform key induces an exact uniform fixed-size subset.
            if criterion == "uniform":
                keys = torch.rand(logits.shape, device=logits.device, dtype=torch.float64, generator=generator)
                rank = _ranks(keys, negative)
            else:
                rank = _ranks(logits, negative, descending=True)
            k = (counts.float() * fraction).round().long().clamp_min(1).minimum(counts)
            keep_negative = negative & (rank < k)
            if criterion == "uniform":
                inclusion = (k / counts.clamp_min(1)).expand_as(logits).clone()
            else:
                # Deterministic truncation has zero support; HT is forbidden below.
                inclusion = keep_negative.float()
        else:
            if criterion == "bernoulli":
                inclusion = torch.full_like(logits, fraction)
            elif criterion == "gradient":
                inclusion = calibrate_inclusion(logits.detach().sigmoid(), negative, fraction)
            else:
                rank = _ranks(logits, negative).float() / (counts - 1).clamp_min(1)
                if criterion == "gaussian":
                    weights = torch.exp(-.5 * ((rank - .8) / .2).square())
                else:
                    bins = (rank * 10).floor()
                    weights = torch.exp(bins * 1.9459101490553132 - 7 - torch.lgamma(bins + 1))
                inclusion = calibrate_inclusion(weights, negative, fraction)
            draws = torch.rand(logits.shape, device=logits.device, generator=generator)
            keep_negative = negative & (draws < inclusion)
        # No forced-nonempty fallback: zero retained negatives is a valid draw.
        return positive | keep_negative, inclusion.masked_fill(positive, 1)


def apply_filter(losses: Tensor, labels: Tensor, keep: Tensor, inclusion: Tensor,
                 objective: str, fraction: float) -> Tensor:
    if objective not in OBJECTIVES:
        raise ValueError(objective)
    if not (losses.shape == labels.shape == keep.shape == inclusion.shape):
        raise ValueError("filter tensors must have identical shapes")
    positive = labels > 0
    negative = ~positive
    if not bool(keep[positive].all()):
        raise ValueError("all positive entries must be retained")
    if objective == "full":
        return losses.mean()
    if objective == "scaled_full":
        return (losses * torch.where(positive, 1., fraction)).sum() / losses.numel()
    if objective == "drop":
        weights = keep.float()
    elif objective == "ht":
        if bool((inclusion[negative] <= 0).any()):
            raise ValueError("HT requires nonzero inclusion probability for every negative")
        weights = keep / inclusion
    else:
        original = negative.sum(dim=0)
        selected = (negative & keep).sum(dim=0)
        scale = (original / selected.clamp_min(1)).expand_as(losses)
        weights = torch.where(positive, torch.ones_like(losses), keep * scale)
    return (losses * weights.detach()).sum() / losses.numel()


def filtering_risk(logits: Tensor, labels: Tensor, criterion: str, fraction: float,
                   objective: str, generator=None) -> tuple[Tensor, dict]:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if criterion == "hard" and objective == "ht" and fraction < 1:
        raise ValueError("a deterministic hard subset cannot estimate the full negative pool by HT")
    if objective in {"full", "scaled_full"}:
        keep, inclusion = torch.ones_like(labels, dtype=torch.bool), torch.ones_like(logits)
    else:
        keep, inclusion = retention_mask(logits, labels, criterion, fraction, generator)
    risk = apply_filter(losses, labels, keep, inclusion, objective, fraction)
    negative = labels == 0
    selected = negative & keep
    counts = selected.sum(dim=0)
    # Tensor diagnostics permit the trainer to defer CPU synchronization to logging.
    info = {"negative_pool": negative.sum().detach(), "retained_negatives": selected.sum().detach(),
            "zero_retained_outputs": (counts == 0).sum().detach(),
            "positive_pool": (labels > 0).sum().detach(), "retained_positives": (keep & (labels > 0)).sum().detach(),
            "inclusion_min": inclusion[negative].min().detach() if negative.any() else logits.new_tensor(1.),
            "per_output_retained": counts.detach()}
    return risk, info
