# CANS accuracy experiment: results and verdict

Date: 2026-11-18. Protocol: `docs/cans_experiment_protocol_20261118.md`.

## Question

Does audit-constrained negative selection (CANS) improve downstream accuracy over uniform negative sampling and hard-negative mining at the same negative budget?

## Setup

- 125 model-eligible frozen ScopeAudit jobs (89 rice, 36 algal), 10% negative budget.
- Final models: logistic, random forest, MLP (fixed ScopeAudit configurations).
- Rules:
  - `uniform`: B uniformly drawn candidates.
  - `hard`: B highest pilot-model positive scores.
  - `zeros_uniform`: B uniformly drawn visible source zeros (clean baseline).
  - `zeros_hard`: B highest pilot scores among visible source zeros.
  - `cans_hard`: B highest pilot scores inside the stratified certified set.
  - `cans_uniform`: B uniform draws inside the certified set.
  - `cans_mix`: half visible source zeros + half certified uniform draws.
  - `cans_adaptive`: if the pooled audit bound is below 1%, use `cans_hard`; otherwise use `cans_mix`.
  - `risk_weighted`: the same selected set as `uniform`, but negative sample weights are `1 - min(0.5, U_j)`.
- Paired job bootstrap intervals (2,000 replicates).

## Results (AP difference versus uniform)

| Source | Model | cans_adaptive | cans_adaptive vs zeros_uniform | zeros_uniform | risk_weighted |
|---|---|---|---|---|---|
| Rice | Logistic | +0.0173 [+0.0031, +0.0340] | -0.0133 [-0.0195, -0.0080] | +0.0094 [+0.0045, +0.0143] | +0.0017 [-0.0011, +0.0048] |
| Rice | Forest | +0.0246 [+0.0126, +0.0372] | -0.0083 [-0.0138, -0.0031] | +0.0094 [+0.0035, +0.0154] | -0.0012 [-0.0049, +0.0024] |
| Rice | MLP | +0.0234 [+0.0100, +0.0384] | -0.0052 [-0.0099, -0.0006] | +0.0094 [+0.0046, +0.0143] | -0.0053 [-0.0104, -0.0011] |
| Alga | Logistic | +0.0842 [+0.0252, +0.1419] | +0.0095 [-0.0263, +0.0496] | +0.0747 [+0.0268, +0.1224] | -0.0619 [-0.1295, +0.0025] |
| Alga | Forest | -0.1396 [-0.2543, -0.0336] | -0.1260 [-0.2410, -0.0235] | -0.0135 [-0.0834, +0.0474] | +0.0101 [-0.0517, +0.0784] |
| Alga | MLP | +0.0708 [+0.0404, +0.0999] | -0.0195 [-0.0509, +0.0133] | +0.0903 [+0.0635, +0.1176] | +0.0029 [+0.0003, +0.0054] |

Top-5% recall:
- Rice: `cans_adaptive` is flat (logistic 0.1023 vs 0.1027; forest 0.0774 vs 0.0774; MLP 0.0762 vs 0.0788) while AP rises.
- Alga: `cans_adaptive` improves logistic (+0.026) and MLP (+0.003), but the forest is negative; `zeros_uniform` improves logistic (+0.018) and MLP (+0.003).
- Brier: `risk_weighted` is worse than uniform in rice for all three models; `cans_adaptive` improves rice Brier for logistic/forest but worsens it in the alga.

## Verdict

**No stable accuracy gain.** The three audit-based designs fail the pre-declared stability criterion:

1. `cans_hard` is positive on the alga and negative on rice.
2. `cans_adaptive` is positive versus uniform on rice for all three models, but it is **worse than the clean-zero baseline** on the jobs where zeros suffice, and the forest model is strongly negative on the alga.
3. `risk_weighted` is null or negative on AP and worsens Brier; the loss correction does not help.

The only consistent positive result is the **clean-zero baseline**: using visible source zeros as negatives improves AP over uniform on rice for all three models (+0.009), and it is mixed on the alga (logistic/MLP positive, forest null). This is not a new algorithm; it is the observed-reference policy already in ScopeAudit, and it is infeasible for 44 of the 89 rice jobs because there are fewer source zeros than the negative budget.

## Interpretation

- Hard-negative mining is harmful under label noise: selecting the most positive-like candidates as negatives amplifies hidden positives. Certification removes the contamination but leaves uninformative hard negatives.
- The audit improves **contamination and coverage**, not the model's discriminative ceiling. The clean-zero pool already captures the useful negative signal; expanding into unknown candidates does not add accuracy.
- The adaptive rule's rice gain over uniform is real but small and is explained by avoiding contaminated unknowns, not by a better negative sample. It is not larger than the clean-zero baseline.
- `risk_weighted` shows that downweighting contaminated negatives is insufficient; the problem is the selected candidates, not only their weights.

## Recommendation

Do not claim an accuracy-improvement contribution from CANS on these tasks. Keep the paper on the audit/benchmark track:

- ScopeAudit quantifies what a negative label licenses, at what coverage and audit cost.
- ProteinGym shows the official leaderboard is a strong safety proxy but not a certificate.
- The plant resources without an official leaderboard are the gap.

If an accuracy paper is still required, the next step is a different research question: a stronger model/feature representation, a task where negative sampling is the measured bottleneck, or a prospective validation. Repeating negative-sampling variants on these tasks is unlikely to produce a stable gain.

## Files

- `all_results_adaptive_10.csv`: 10% budget, uniform/zeros_uniform/cans_adaptive, three models.
- `all_results_risk_weighted.csv`: 10% budget, uniform/risk_weighted, three models.
- `summary_adaptive_10.csv`, `summary_risk_weighted.csv`: paired contrasts.
- `all_results_adaptive001_lr.csv`: earlier logistic-only adaptive run.

## RCSS: risk-constrained stratified sampler (89 rice tasks)

RCSS solves
`max sum_j b_j * hardness_j s.t. sum_j b_j = B, sum_j b_j U_j <= epsilon B, 0 <= b_j <= N_j`
with a small linear program and selects the hardest `b_j` candidates per stratum.

On the 52 rice jobs where CANS and RCSS are common-feasible at epsilon = 0.10:

| Rule | ΔAP logistic | ΔAP forest | ΔAP MLP | Realized b_+ | Audit |
|---|---|---|---|---|---|
| CANS-hard | -0.057 | -0.007 | -0.023 | 1.93% | 10% |
| CANS-mix | +0.004 | +0.021 | +0.021 | 0.32% | 10% |
| RCSS e10 | -0.058 | -0.013 | -0.014 | 2.18% | 10% |
| Zeros (45 jobs) | +0.004 | +0.006 | +0.011 | 0% | 0 |

RCSS satisfies the risk bound (mean LP bound 0.095; realized contamination 2.18%), but it is not better than CANS-hard and is worse than the zero-anchored CANS-mix and the clean-zero rule. The risk constraint is not the bottleneck; the hard-negative objective is. RCSS is a negative result for accuracy and a supporting result for the paper's main claim: the audit's value is risk control, not a better hard-negative sampler.
