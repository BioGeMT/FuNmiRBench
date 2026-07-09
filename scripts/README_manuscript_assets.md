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

Optional Figure 1 panel D gene-length input:

```bash
uv run scripts/build_manuscript_figures_tables.py \
  --report-dir results/20260703_115539 \
  --out-dir manuscript_assets \
  --gene-lengths-tsv data/resources/gene_lengths/ensembl_v115_utr3_lengths.tsv
```

The gene-length table may be TSV or CSV and should contain a gene identifier column plus a 3'UTR length column. Preferred schema:

```text
gene_id    utr3_length_bp
ENSG...    1234
```

Accepted gene ID aliases include `gene_id`, `Ensembl_ID`, `ensembl_gene_id`, `ensembl_id`, `gene`, and `GeneID`. Accepted length aliases include `utr3_length_bp`, `three_prime_utr_length`, `three_prime_utr_length_bp`, `3utr_length`, `3utr_length_bp`, `utr3_len`, `utr3_len_bp`, `length_bp`, and `length`.

## Main manuscript outputs

Figures:

- `figures/figure1_cross_dataset_distributions.png` and `.svg`
- `figures/figure1_panel_c_gene_universes.png` and `.svg`
- `figures/figure1_panel_d_gene_lengths.png` and `.svg` when `--gene-lengths-tsv` is provided
- `figures/figure2_rank_enrichment_recovery.png` and `.svg`
- `figures/figure3_targetscan_centered.png` and `.svg`

Tables:

- `tables/table1_cross_dataset_predictor_summary.tsv`
- `tables/figure1_panel_c_gene_universe_counts.tsv`
- `tables/figure1_panel_d_gene_lengths.tsv` when `--gene-lengths-tsv` is provided
- `tables/figure1_panel_d_gene_length_qc.tsv` when `--gene-lengths-tsv` is provided

## Supplementary output

Tables:

- `tables/table_s1_per_dataset_predictor_metrics.tsv`

## Input assumptions

The report directory should contain:

- cross-dataset metric tables such as `coverage_per_experiment.tsv`, `positive_coverage_per_experiment.tsv`, `aps_per_experiment.tsv`, `pr_auc_per_experiment.tsv`, `auroc_per_experiment.tsv`, and `spearman_per_experiment.tsv`
- per-dataset `joined.tsv` files with GT columns (`logFC`, `FDR`) and predictor score/rank columns

The GT-positive rule used by the post-processing figures is the manuscript rule: `FDR < 0.05` and perturbation-aware expected effect `> 1.0` by default.

`figure1_panel_c_gene_universes` is a schematic of the predictor-scored gene-set overlap. It labels the full gene set (FGS) as the full usable joined-table universe and the intersection gene set (IGS) as genes scored by every selected predictor. The accompanying TSV stores the per-dataset FGS, IGS, union-scored, and per-predictor scored-gene counts.

`figure1_panel_d_gene_lengths` is a mirrored density plot comparing 3'UTR lengths for IGS genes versus non-IGS genes. By default it uses one row per dataset-gene pair because IGS membership is miRNA/dataset-specific. For a sensitivity-style collapsed plot, pass `--panel-d-membership-mode unique_gene`.
