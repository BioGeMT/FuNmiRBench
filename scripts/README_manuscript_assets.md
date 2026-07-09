# Manuscript figure/table asset builder

This branch contains post-processing code for the figures and tables used in the functional miRNA predictor manuscript. It does **not** rerun the benchmark; it consumes an already generated FuNmiRBench report directory.

The current manuscript plan keeps the supplement minimal: **only the per-dataset predictor metrics table is included as supplementary material**. Extra per-dataset/per-predictor plots are intentionally not generated here.

## Script

```bash
python scripts/build_manuscript_figures_tables.py \
  --report-dir results/20260703_115539 \
  --out-dir manuscript_assets
```

Optional thresholds:

```bash
python scripts/build_manuscript_figures_tables.py \
  --report-dir results/20260703_115539 \
  --out-dir manuscript_assets \
  --fdr-threshold 0.05 \
  --effect-threshold 1.0
```

## Main manuscript outputs

Figures:

- `figures/figure1_cross_dataset_distributions.png` and `.svg`
- `figures/figure1_panel_c_gene_universes.png` and `.svg`
- `figures/figure2_rank_enrichment_recovery.png` and `.svg`
- `figures/figure3_targetscan_centered.png` and `.svg`

Tables:

- `tables/table1_cross_dataset_predictor_summary.tsv`
- `tables/figure1_panel_c_gene_universe_counts.tsv`

## Supplementary output

Tables:

- `tables/table_s1_per_dataset_predictor_metrics.tsv`

## Input assumptions

The report directory should contain:

- cross-dataset metric tables such as `coverage_per_experiment.tsv`, `positive_coverage_per_experiment.tsv`, `aps_per_experiment.tsv`, `pr_auc_per_experiment.tsv`, `auroc_per_experiment.tsv`, and `spearman_per_experiment.tsv`
- per-dataset `joined.tsv` files with GT columns (`logFC`, `FDR`) and predictor score/rank columns

The GT-positive rule used by the post-processing figures is the manuscript rule: `FDR < 0.05` and perturbation-aware expected effect `> 1.0` by default.

`figure1_panel_c_gene_universes` is a schematic of the predictor-scored gene-set overlap. It labels the full gene set (FGS) as the full usable joined-table universe and the intersection gene set (IGS) as genes scored by every selected predictor. The accompanying TSV stores the per-dataset FGS, IGS, union-scored, and per-predictor scored-gene counts.
