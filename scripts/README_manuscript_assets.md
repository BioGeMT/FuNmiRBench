# Manuscript figure/table asset builder

This branch contains post-processing code for the figures and tables used in the functional miRNA predictor manuscript. It does **not** rerun the benchmark; it consumes an already generated FuNmiRBench report directory.

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
- `figures/figure2_rank_enrichment_recovery.png` and `.svg`
- `figures/figure3_targetscan_centered.png` and `.svg`

Tables:

- `tables/table1_cross_dataset_predictor_summary.tsv`

## Supplementary outputs

Figures:

- `figures/supplementary_mirdb_centered.png` and `.svg`

Tables:

- `tables/table_s1_detailed_cross_dataset_predictor_summary.tsv`
- `tables/table_s2_targetscan_centered_bins.tsv`
- `tables/table_s3_mirdb_centered_bins.tsv`
- `tables/table_s4_local_rank_bins.tsv`
- `tables/table_s5_global_rank_bins.tsv`
- `tables/table_s6_recovery_curves.tsv`
- `tables/per_experiment_metric_rows.tsv`

## Input assumptions

The report directory should contain:

- cross-dataset metric tables such as `coverage_per_experiment.tsv`, `positive_coverage_per_experiment.tsv`, `aps_per_experiment.tsv`, and `auroc_per_experiment.tsv`
- per-dataset `joined.tsv` files with GT columns (`logFC`, `FDR`) and predictor score/rank columns

The GT-positive rule used by the post-processing figures is the manuscript rule: `FDR < 0.05` and perturbation-aware expected effect `> 1.0` by default.
