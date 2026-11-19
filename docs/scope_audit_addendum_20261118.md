# ScopeAudit certification addendum (2026-11-18)

## Timing

The main protocol (`docs/scope_audit_protocol_20261118.md`) registered stratified
certification with a Bonferroni-corrected Clopper–Pearson bound per stratum.
After a two-job debug run (rice target 0, seed 1907, hiding 0.50; alga target 1,
seed 1907, hiding 0.50), the per-stratum bound was visibly conservative: with
about eight audits per stratum and twelve strata, a stratum with zero observed
hidden positives still had an upper bound above 0.30. The aggregate
certification results had not been examined at that point.

Before the full analysis, a second certification mode was added: a single
pooled Clopper–Pearson bound over all audited candidates in a job. If the
pooled bound is at most epsilon, the entire candidate pool is certified.
This mode uses the same audit samples, strata, audit fractions, epsilons and
delta as the registered analysis. It does not change the registered jobs,
rules, hypotheses, metrics or thresholds. It adds a coarser whole-pool
certificate alongside the partial stratified certificate.

## Consequence for interpretation

- The pooled mode answers: "can the whole candidate pool be used at tolerance
  epsilon?" Its coverage is 100% or 0%.
- The stratified mode answers: "which observable strata can be used at
  tolerance epsilon?" Its coverage is partial and its bound is corrected for
  the number of strata.
- Neither mode certifies a natural unknown pool. Both are conditional on
  random sampling within strata and exchangeability of checked and unchecked
  candidates within an accepted stratum.

## High-assurance downstream sensitivity

The primary downstream certification point (a = 0.05, epsilon = 0.05) certifies
no rice jobs under the Bonferroni-corrected stratified rule. Before examining
the new downstream predictions, a second, pre-declared high-assurance point
(a = 0.20, epsilon = 0.10) was added. It is reported separately, labelled
`scope_audit_high`, and is not substituted for the primary point or used to
select a favorable result.

The full run was restarted after this change. The original registration file
was retained: it records the registered job list, budgets, policies, audit
fractions, epsilons and protocol hash. The code hash in the final registration
revision is recorded in `results/scope_audit_v1/registration_revision.json`.
