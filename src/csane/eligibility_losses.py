"""Binary development controls. PU risk correction is not per-item admission."""
import torch
from torch.nn import functional as F


def selected_bce(logits, positive, candidates, selected, negative_weight=1.):
    """Keep full candidate-negative mass, avoiding a count-induced weight change."""
    if torch.any(selected & (~candidates | positive)):
        raise ValueError("negative supervision crossed an admissibility boundary")
    n = positive.sum() + candidates.sum()
    if n == 0 or not positive.any() or not selected.any():
        raise ValueError("this pilot needs observed positives and a nonempty selected pool")
    return (F.softplus(-logits[positive]).sum() +
            negative_weight * candidates.sum() * F.softplus(logits[selected]).mean()) / n


def asymmetric_loss(logits, positive, admissible, gamma_negative=4., clip=.05):
    """ASL-style binary control, gamma-positive=0; retain focusing-weight gradient."""
    if not 0 <= clip < 1 or gamma_negative < 0:
        raise ValueError("invalid ASL parameters")
    p = logits.sigmoid()
    negative_probability = (1 - p + clip).clamp(max=1.)
    positive_terms = -torch.log(p[positive].clamp(min=1e-8)).sum()
    negative = admissible & ~positive
    negative_terms = -((1 - negative_probability[negative]) ** gamma_negative *
                       torch.log(negative_probability[negative].clamp(min=1e-8))).sum()
    return (positive_terms + negative_terms) / admissible.sum()


def nnpu_objective(logits, positive, population, prior, beta=0., gamma=1.):
    """Kiryo et al. Eq.6 value and Algorithm1 update (logistic loss).

    U is the complete source-admissible training comparison population, including
    visible positives. It need not be the whole measured assay population: e.g.
    source-excluded DepMap intermediate effects are outside this binary domain.
    P/U therefore overlap in this finite masked-label pilot; do not import the
    independent-sample theorem. `prior` comes from a declared simulation rule.
    Return correction-branch objective plus descriptive risk components.
    """
    if not 0 < prior < 1 or beta < 0 or not 0 <= gamma <= 1:
        raise ValueError("invalid nnPU prior/correction parameters")
    if not positive.any() or not population.any() or torch.any(positive & ~population):
        raise ValueError("invalid P/U domains")
    positive_risk = prior * F.softplus(-logits[positive]).mean()
    negative_risk = F.softplus(logits[population]).mean() - prior * F.softplus(logits[positive]).mean()
    corrected = bool(negative_risk.detach() < -beta)
    update = -gamma * negative_risk if corrected else positive_risk + negative_risk
    estimated = positive_risk + negative_risk.clamp(min=0)
    return update, dict(positive_risk=positive_risk.detach(), negative_risk=negative_risk.detach(),
                        estimated_risk=estimated.detach(), correction_branch=corrected)
