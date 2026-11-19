# P1 report: ProteinGym negative-label safety

Date: 2026-11-18. Frozen protocol: `docs/p1_audit_protocol_20261118.md`.

## What was run

- ProteinGym v1.3 DMS substitutions: 217 assays.
- Official v1.3 zero-shot score archive: 95 model columns; 89 models matched to an official leaderboard rank.
- Official leaderboard: `Summary_performance_DMS_substitutions_Spearman.csv`; per-assay Spearman reference: `DMS_substitutions_Spearman_DMS_level.csv`.
- Negative-label audit: for each assay, select the bottom q% of a model's score as the negative set (q = 5, 10, 20%); define high-fitness variants as the top 25% of measured fitness; report selected-set contamination b_+, hidden-positive selection e_H, and judged contamination c_J.
- Eight of the official top-10 models are present in the public score archive. AIDO Protein-RAG (rank 1) and Protriever (rank 8) are absent; ProSST, PoET, ESM3 and SaProt names were matched by alias. The comparison therefore uses the official top-10 *available* models.

## Main result

The official ProteinGym leaderboard is a strong proxy for negative-label safety, not a certificate.

| Budget q | Spearman ρ (official rank vs b_+) | R² (linear) | Top-10 available mean b_+ | Rest mean b_+ | Top-10 above 5% | All models above 5% |
|---|---|---|---|---|---|---|
| 5% | -0.90 | 0.82 | 3.09% | 8.17% | 12.5% | 82.0% |
| 10% | -0.94 | 0.90 | 3.91% | 8.81% | 25.0% | 91.0% |
| 20% | -0.97 | 0.95 | 5.72% | 10.24% | 75.0% | 97.8% |

- H1 (official rank is not strongly related to b_+) is rejected: ρ = -0.94 at q = 10%.
- H2 (top-10 models are not safer than the rest) is rejected: Mann–Whitney p = 1.4 × 10⁻⁹ at q = 10%.
- H4 (a top-10 model is worse than the median outside the top 10) is rejected: the worst top-10 available model is ESCOTT at 5.65%, below the rest median.
- The leaderboard still does not certify a tolerance. At q = 20%, six of the eight top-10 available models exceed 5% contamination; only VenusREM (4.84%) and ProSST K=2048 (4.87%) are below it. At q = 10%, two of eight exceed 5% (S2F-MSA 5.21%, ESCOTT 5.65%).

## Top-10 available models at q = 10%

| Rank | Model | Official Spearman | Mean b_+ | Mean e_H |
|---|---|---|---|---|
| 2 | VenusREM | 0.518 | 3.25% | 1.28% |
| 3 | ProSST (K=2048) | 0.507 | 3.29% | 1.29% |
| 4 | ProSST (K=4096) | 0.498 | 2.82% | 1.12% |
| 5 | S3F-MSA | 0.496 | 4.92% | 1.94% |
| 6 | S2F-MSA | 0.488 | 5.21% | 2.06% |
| 7 | ProSST (K=1024) | 0.485 | 2.95% | 1.17% |
| 9 | ESCOTT | 0.476 | 5.65% | 2.23% |
| 10 | ProSST (K=512) | 0.471 | 3.15% | 1.24% |

## Residual structure

The linear fit leaves model-family structure:

- Worse than rank predicts: Site-Independent (rank 76, b_+ 14.2% at q = 10%, +4.4 points above prediction), xTrimoPGLM-7B-CLM (rank 97, 22.7%, +2.5 points), DeepSequence (single/ensemble) and EVmutation (+1.9 to +2.1 points).
- Better than rank predicts: ProSST K=20, K=512, K=128 and K=1024 (mean b_+ 3.0–4.0%, 1.7–2.6 points below prediction), and ProteinMPNN (rank 91, b_+ 12.1%, 1.8 points below prediction, but still high in absolute terms).

This means the official average Spearman is not a sufficient statistic for tail safety across model families; two models with similar average rank can differ by several percentage points of contamination.

## What this changes for the paper

The original P1 hypothesis ("top official models still have unsafe negatives") is only partly supported. The correct claim is:

> Where an official leaderboard exists, it ranks negative-label safety surprisingly well, but it does not certify it. On ProteinGym, rank explains 90–95% of between-model contamination variance; nevertheless, 25% of top-10 models exceed a 5% tolerance at q = 10% and 75% at q = 20%. The remaining variance is model-family structured. The leaderboard reports neither the tail metric nor a tolerance.

This is the "good case." The plant resources without an official leaderboard have no such proxy at all; the cross-resource gap is the paper's main target, not a claim that ProteinGym's leaderboard is broken.

## Next

- P1b: CAFA6/GO. The official CAFA6 Kaggle data and GOAlpha weights are not directly accessible; the reproducible path is the plant GO benchmark (DeepGOPlus/DeepGreenGO) on the Arabidopsis/rice annotations already in this repository, with the CAFA6 leaderboard reported as context. If the Kaggle data can be obtained, add the official top-k submissions.
- P2: localization, plant PPI and TF binding, using the 2025–2026 published benchmarks identified in the resource inventory.

## Files

- `assay_audit.csv`: 61,848 assay × model × budget rows.
- `assay_audit_with_metadata.csv`: with ProteinGym assay type and MSA-depth metadata.
- `model_summary.csv`: 89 ranked models × 3 budgets.
- `p1_summary.json`: correlations, top-10 comparisons, residuals, assay-type and MSA-depth strata.
- `figures/fig_p1_proteingym.pdf/png`: official Spearman vs contamination at q = 10% and q = 20%.

## Sensitivity: top-10% high-fitness definition

The protocol pre-declared the top 25% measured-fitness definition as primary and the top 10% as a sensitivity analysis. Under the stricter definition:

| Budget q | Spearman ρ | Top-10 available mean b_+ | Rest mean b_+ |
|---|---|---|---|
| 10% | -0.93 | 1.53% | 3.68% |
| 20% | -0.95 | 2.25% | 4.26% |

The rank-proxy conclusion is robust. Absolute contamination is lower because the high-fitness set is smaller and more extreme. Results are in `high90/`.
