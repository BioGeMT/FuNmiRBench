# Figure 2 Coverage Denominator Check

## Why TargetScan Can Have 67.3% Gene Coverage But 12-13% Pair Coverage

| Metric | Denominator | Numerator | TargetScan value | Interpretation |
|---|---:|---:|---:|---|
| Panel B gene-set coverage | 19,540 unique benchmark genes | 13,146 genes with at least one TargetScan score | 67.3% | Gene-level coverage asks whether a gene is covered by TargetScan for any benchmark miRNA context. |
| Panel C positive coverage | 12,837 perturbation-consistent positive miRNA-gene pairs | 1,593 positive pairs with a TargetScan score | 12.4% | Positive pair coverage asks whether TargetScan scores the specific perturbed miRNA-gene pair in each experiment. |
| Panel D background coverage | 496,429 background/non-positive miRNA-gene pairs | 65,791 background pairs with a TargetScan score | 13.3% | Background pair coverage uses the same miRNA-specific pair denominator for non-positive genes. |
| All usable benchmark pair coverage | 509,266 benchmark miRNA-gene pairs | 67,384 pairs with a TargetScan score | 13.2% | Across all usable benchmark pairs, TargetScan scores only a minority of the specific miRNA-gene combinations. |

Manuscript wording:

> TargetScan covered 67.3% of the full benchmark gene set at the gene level, but only 12.4% of positive and 13.3% of background benchmark miRNA-gene pairs. This reflects the distinction between gene coverage and miRNA-specific pair coverage: many genes have at least one TargetScan prediction for some miRNA, but the perturbed benchmark miRNA is not predicted for most of those genes.

## Supplementary Table

The complete numerator/denominator values for Figure 2, including the mean number of benchmark miRNAs per scored gene for each predictor, are in `figure2_supplementary_coverage_table.tsv`.

TargetScan-specific sanity check: among genes scored by TargetScan in the usable benchmark rows, the mean number of benchmark miRNAs covered per gene is 4.62.
