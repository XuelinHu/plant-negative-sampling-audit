# ScopeAudit protocol (frozen 2026-11-18)

## Working title

ScopeAudit: a risk–coverage framework for auditing negative labels in plant functional data

## Motivation

Plant functional resources encode three different objects in the same "negative" state: an untested relationship, a tested relationship that was not detected in one context, and a relationship that is contradicted by a matched positive record. Downstream users then treat the state as if it licensed exclusion or as if it were a clean training negative. Positive–unlabeled (PU) learning already treats unlabeled data as a mixture and does not require unlabeled examples to be negative. What is missing is a plant-specific audit that (i) states the scope of a negative claim, (ii) estimates the contamination of a sampling rule on the same candidate pool, and (iii) certifies a subset of negatives at a stated error tolerance with an explicit audit cost.

ScopeAudit is that audit. It is not a new theorem about contraposition; contraposition and modus tollens are the logical warrant for asking for a counterexample. The contribution is the estimand, the certification procedure, the risk–coverage frontier, and the benchmark.

## Formal objects

A negative label is a scoped claim

    r = (s, i, t, c, m, tau)

with source s, entity i, target t, biological context c, measurement m and information time tau. A claim H_S asserts that, within scope S, entity i has no positive relationship to target t. The scope must state the acceptable contamination rate epsilon and the confidence 1 - delta; without them, "negative" is not a decision.

For a candidate pool C_t (training entities whose target is not a visible positive), a sampling rule pi selects E_t subset C_t of size B_t. Hidden positives H_t subset C_t are source-positive records that the operational label treats as negative. The audit estimands are:

- hidden-positive selection rate: e_H = |E_t ∩ H_t| / |H_t|;
- sampled-set contamination: b_+ = |E_t ∩ H_t| / |E_t|;
- judged contamination: c_J = |E_t ∩ H_t| / (|E_t ∩ H_t| + |E_t ∩ Z_t|), where Z_t are visible source zeros;
- coverage: kappa = |E_t| / |C_t|;
- feasibility: |E_t| >= B_t.

e_H is a recall-like quantity. b_+ and c_J are contamination-like quantities. They answer different questions and are always reported together.

## Hypotheses

- H1 (frontier): no tested rule achieves both b_+ <= 0.05 and kappa >= 0.50 across rice and algal sources at the registered budgets.
- H2 (heuristic cost): context deferral reduces b_+ relative to uniform only at a coverage cost, and the reduction is not uniform across sources or budgets.
- H3 (certification): stratified ScopeAudit certification yields a certified set whose realized contamination is at most epsilon and whose empirical bound coverage is at least 1 - delta, at a lower audit cost than checking every selected label.
- H4 (downstream separation): lower b_+ does not automatically increase AP or top-5% recovery of measured positives; PU learning is a complementary baseline, not a replacement for certification.
- H5 (natural audits): the same risk–coverage trade-off appears in annotation time splits and cross-context phenotype/binding records.

## Data and admission

Reuse the frozen cache at data/processed/contrapositive_sampling_v1:

- rice: 2,878 lines x 9 traits; source zero = registered normal; source one = registered abnormal; unknown = missing/conflicting.
- alga: 15,582 genes x 223 screens (222 metadata matched); source positive = finite FDR <= 0.05; source zero = larger finite value; unknown = unavailable.
- annotation cohorts: 1,224 all-three-unknown and 4,598 MF/BP-unknown Arabidopsis loci.
- binding: 901 call sets from three land-plant species.

The six rice traits and three algal targets that pass the stricter model gate are unchanged. Admission is fixed before the new outcomes are computed.

## Controlled hiding

For each source, eligible target and seed in {1907, 2909, 3911, 4919, 5923}:

- split entities by a 70/30 hash of the entity group;
- hide a fraction h in {0.25, 0.50, 0.75} of the training source positives (at least one);
- set hidden cells to unknown in the visible matrix;
- candidate pool C = training entities with visible state != positive;
- budgets gamma in {0.05, 0.10, 0.20, 0.30, 0.50} of |C|;
- test entities remain untouched and are never used by a rule.

The hiding fraction and budget grid are registered here, before the new runs.

## Rules

1. uniform: every candidate is eligible; draw B uniformly without replacement.
2. context_deferred: candidate must have at least one other measured context and no other-context source-positive input.
3. matched_control: stratified random sample matching the context-deferred count within coverage tertiles and visible state (measured zero/unknown).
4. observed_reference: visible source zeros only; zero hidden exposure is by construction and is not an independent safety certificate.
5. ScopeAudit certification: see below.

Every draw is without replacement. An insufficient pool is infeasible; the budget is not reduced and is not filled by repeated draws. Hidden labels are accessed only after selection, for evaluation.

## ScopeAudit certification

For each job, partition C into observable strata:

    coverage_bin = tertile of the number of other measured contexts
    visible_state = measured zero vs unknown
    other_positive = any other-context source positive
    stratum = 4 * coverage_bin + 2 * visible_state + other_positive

This gives at most 12 strata and uses no hidden label. The strata are fixed a priori.

Audit budget a in {0.02, 0.05, 0.10, 0.20} of |C|. Allocate the audit sample proportionally to stratum size with at least one candidate per non-empty stratum; draw uniformly without replacement within each stratum. Checking a candidate reveals whether it is a hidden positive.

For stratum j with n_j checked and a_j hidden positives, the one-sided Clopper–Pearson upper bound at level alpha_j is

    U_j = Beta^{-1}(1 - alpha_j; a_j + 1, n_j - a_j),

with U_j = 1 when a_j = n_j. To control the familywise error over the strata of one job at delta, set alpha_j = delta / m, where m is the number of non-empty strata (Bonferroni). A stratum is accepted when U_j <= epsilon. The certified set is the union of accepted strata. Its realized contamination is evaluated with the hidden labels but is not used by the rule.

The certification guarantee is conditional on random sampling within strata and exchangeability of checked and unchecked candidates within a stratum. It is a bound on contamination of the source-positive subset, not on an unknown natural pool.

A second, assumption-free comparator audits exactly the selected set and removes the checked hidden positives. Its certified set is the checked non-hidden candidates; its audit cost equals its selected size. The comparison asks whether the exchangeability assumption buys coverage at the same audit cost.

Certification is repeated 20 times per job and audit budget to estimate calibration. Calibration is the fraction of strata in which U_j >= the true stratum contamination p_j. The procedure is calibrated when this fraction is at least 1 - delta within Monte Carlo error.

## Metrics and aggregation

For uncertified rules: e_H, b_+, c_J, kappa, feasibility, selected composition (hidden positive / source zero / natural unknown).

For ScopeAudit: audit cost, certified coverage, certified realized b_+, number of accepted strata, maximum and mean U_j, calibration, and the assumption-free comparator at the same audit cost.

Aggregation uses the exact common-feasible jobs for a paired comparison. Intervals are 1,000-replicate common-feasible job bootstrap percentiles, preserving target and split composition. They are unadjusted across contrasts and conditional on the observed sources and hiding design; they are not familywise confidence intervals and not biological replication.

The risk–coverage frontier is the set of achievable (b_+, kappa) points by rule and budget. A point is Pareto-efficient if no other registered point has both lower b_+ and higher kappa in the same source and hiding fraction. Lines between budgets are descriptive because job composition can change.

## Downstream prediction

Downstream prediction is evaluated at gamma in {0.10, 0.30} and the full-observed reference. The primary certification operating point for the downstream comparison is a = 0.05 and epsilon = 0.05. A pre-declared high-assurance sensitivity uses a = 0.20 and epsilon = 0.10; it is reported separately and is not substituted for the primary point.

The model gate is unchanged: at least five visible training positives, at least five test positives, at least five test zeros and model-eligible metadata. Standardize features on training entities only.

Models: L2 logistic (C = 1, L-BFGS, 250 iterations), random forest (48 trees, depth 10, leaf 5) and MLP (32 ReLU units, Adam 0.001, 60 epochs, tolerance 1e-4, 10 non-improving epochs). Negative mass matches the visible source-zero count. No test-based selection.

PU baselines: uPU and nnPU linear logistic on the same features and splits, using all unlabeled training candidates, with the Elkan–Noto class-prior estimate from five-fold cross-fitted logistic P-vs-U probabilities. Fixed 500 L-BFGS iterations; no test-based selection. A sensitivity analysis uses the observed positive fraction as the prior.

Metrics: AP, ROC-AUC, Brier, top-5% measured-positive recall. Prediction contrasts use pairwise-matched endpoint–seed sets and seed macros. Reports include all MLP convergence warnings.

## Natural audits

- Annotation: reuse the frozen 2023 unknown cohorts and current aspect-matched primary assertions. The negative claim is "unknown for aspect A"; the counterexample is a current aspect-matched assertion. Report the selected-set contradiction lower bound and the risk–coverage trade-off at the registered draws (uniform, historical deferral, history-matched control; 5 seeds; budgets 0.1/0.3/0.5).
- Cross-context: rice trait pairs and algal screen pairs; the negative claim is "source-normal in context A excludes abnormal in context B"; report the transfer counterexample fraction and its denominator.
- Binding: ChIP-Hub call pairs within source study/factor groups; the negative claim is "no call in study A excludes a call in study B"; report non-recovery medians and pair counts. This is call portability, not biological inactivity.

## Decision rule

For a stated use, tolerance epsilon and confidence 1 - delta:

1. compute the risk–coverage frontier for the registered rules;
2. discard rules whose certified or realized b_+ exceeds epsilon;
3. among the remainder, choose the rule with the largest coverage;
4. if no rule remains, report "no certified negative set" and defer the measurement; do not relabel the candidates as negative.

The recommendation is conditional. It does not license all-plant inference, natural-pool purity, causal function, or prospective discovery.

## Registration

The code snapshot, configuration, source hashes and protocol hash are written to results/scope_audit_v1/registration.json before the first new job. All registered outcomes are retained, including infeasible budgets, null contrasts and adverse model results.
