# Manuscript figure/table asset builder

This branch contains post-processing code for the figures and tables used in the functional miRNA predictor manuscript. It does not rerun the benchmark; it consumes an already generated FuNmiRBench report directory.

The current manuscript plan keeps the supplement minimal: only the per-dataset predictor metrics table is included as supplementary material. Extra per-dataset/per-predictor plots are intentionally not generated here.

## Script

Run the main manuscript asset builder:

```bash
python scripts/build_manuscript_figures_tables.py --report-dir results/20260703_115539 --out-dir manuscript_assets
```

Run the top-500 recovery version for Figure 2:

```bash
python scripts/build_manuscript_figures_tables_top500.py --report-dir results/20260706_132519 --out-dir manuscript_assets
```

Run the corrected reference-centered Figure 3 builder:

```bash
python scripts/build_figure3_targetscan_centered.py --report-dir results/20260706_132519 --out-dir manuscript_assets
```

Run the exploratory gene-level predictability analysis:

```bash
python scripts/build_gene_predictability.py --report-dir results/20260706_132519 --out-dir manuscript_assets --min-positive-occurrences 2
```

Optional thresholds can be provided with `--fdr-threshold 0.05` and `--effect-threshold 1.0`.

## Main manuscript outputs

Figures:

- `figures/figure1_cross_dataset_distributions.png` and `.svg`
- `figures/figure2_rank_enrichment_recovery.png` and `.svg`
- `figures/figure3_reference_centered.png` and `.svg`

Figure 2 is now a three-panel local-rank figure:

- local rank-bin enrichment
- local rank distributions for background versus GT-positive miRNA-gene pairs
- fixed-budget GT-positive recovery

The global-rank enrichment panel is no longer part of the main Figure 2. Use the top-500 wrapper when the recovery panel should extend to 500 admitted predictions per dataset.

Figure 3 is a reference-rank analysis with TargetScan-centered and miRDB-centered rows in the same figure. For each reference predictor, the builder recomputes the reference local rank within each dataset from the reference score using average ranks for ties, bins reference-scored miRNA-gene pairs, and then calculates enrichment and cross-predictor coverage inside those same reference-defined bins. The active reference predictor keeps its predictor color and is drawn as a dashed line.

Tables:

- `tables/table1_cross_dataset_predictor_summary.tsv`

## Supplementary and exploratory outputs

Tables:

- `tables/table_s1_per_dataset_predictor_metrics.tsv`
- `tables/table_s2_local_rank_background_positive_summary.tsv`
- `tables/figure3_reference_centered_summary.tsv`
- `tables/gene_predictability_rank_occurrences.tsv`
- `tables/gene_predictability_by_predictor.tsv`
- `tables/gene_predictability_overall.tsv`

Exploratory figures:

- `figures/gene_predictability_scatter.png` and `.svg`
- `figures/gene_predictability_heatmap.png` and `.svg`

The gene-level predictability analysis asks whether some genes are recurrently easier to prioritize as GT-positive miRNA targets. It aggregates local ranks by gene and predictor across datasets, focusing on occurrences where the gene is GT-positive for the perturbed miRNA.

## Input assumptions

The report directory should contain:

- cross-dataset metric tables such as `coverage_per_experiment.tsv`, `positive_coverage_per_experiment.tsv`, `aps_per_experiment.tsv`, `pr_auc_per_experiment.tsv`, `auroc_per_experiment.tsv`, and `spearman_per_experiment.tsv`
- per-dataset `joined.tsv` files with GT columns (`logFC`, `FDR`) and predictor score/rank columns

The GT-positive rule used by the post-processing figures is the manuscript rule: FDR < 0.05 and perturbation-aware expected effect > 1.0 by default.
