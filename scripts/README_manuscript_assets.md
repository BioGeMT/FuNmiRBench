# Manuscript figure/table asset builder

This branch contains post-processing code for the figures and tables used in the functional miRNA predictor manuscript. It does **not** rerun the benchmark; it consumes an already generated FuNmiRBench report directory.

The current manuscript plan keeps the supplement minimal: **only the per-dataset predictor metrics table is included as supplementary material**. Extra per-dataset/per-predictor plots are intentionally not generated here.

## Script

```bash
uv run scripts/build_manuscript_figures_tables.py \
  --report-dir results/20260703_115539 \
  --out-dir manuscript_assets
```

Optional thresholds:

```bash
uv run scripts/build_manuscript_figures_tables.py \
  --report-dir results/20260703_115539 \
  --out-dir manuscript_assets \
  --fdr-threshold 0.05 \
  --effect-threshold 1.0
```

Panel D does not require an extra input file. The script derives 3'UTR lengths from the cached Ensembl release 115 GTF used by the benchmark (`data/resources/ensembl/Homo_sapiens.GRCh38.115.gtf.gz`). If the GTF is missing, the shared Ensembl-resource helper downloads it and caches `data/resources/ensembl/utr3_lengths.tsv` for future manuscript builds.

Panel D uses one row per dataset-gene pair by default, because IGS membership is miRNA/dataset-specific. To collapse to one row per unique gene for a sensitivity-style version:

```bash
uv run scripts/build_manuscript_figures_tables.py \
  --report-dir results/20260703_115539 \
  --out-dir manuscript_assets_unique_genes \
  --panel-d-membership-mode unique_gene
```

## Main manuscript outputs

Figures:

- `figures/figure1_cross_dataset_distributions.png` and `.svg`
- `figures/figure1_panel_c_gene_universes.png` and `.svg`
- `figures/figure1_panel_d_gene_lengths.png` and `.svg`
- `figures/figure2_rank_enrichment_recovery.png` and `.svg`
- `figures/figure3_targetscan_centered.png` and `.svg`

Tables:

- `tables/table1_cross_dataset_predictor_summary.tsv`
- `tables/figure1_panel_c_gene_universe_counts.tsv`
- `tables/figure1_panel_d_gene_lengths.tsv`
- `tables/figure1_panel_d_gene_length_qc.tsv`

## Supplementary output

Tables:

- `tables/table_s1_per_dataset_predictor_metrics.tsv`

## Input assumptions

The report directory should contain:

- cross-dataset metric tables such as `coverage_per_experiment.tsv`, `positive_coverage_per_experiment.tsv`, `aps_per_experiment.tsv`, `pr_auc_per_experiment.tsv`, `auroc_per_experiment.tsv`, and `spearman_per_experiment.tsv`
- per-dataset `joined.tsv` files with GT columns (`logFC`, `FDR`) and predictor score/rank columns

The GT-positive rule used by the post-processing figures is the manuscript rule: `FDR < 0.05` and perturbation-aware expected effect `> 1.0` by default.

`figure1_panel_c_gene_universes` is a schematic of the predictor-scored gene-set overlap. It labels the full gene set (FGS) as the full usable joined-table universe and the intersection gene set (IGS) as genes scored by every selected predictor. The accompanying TSV stores the per-dataset FGS, IGS, union-scored, and per-predictor scored-gene counts.

`figure1_panel_d_gene_lengths` is a mirrored density plot comparing computed Ensembl 3'UTR lengths for IGS genes versus non-IGS genes. The accompanying QC table reports length-match fractions, unique-gene counts, and median/mean 3'UTR lengths for both groups.
