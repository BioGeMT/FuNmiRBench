# Manuscript Figure Scripts

This folder contains scripts used to generate manuscript figure assets and the
tables that support them.

## Output Layout

Stable manuscript outputs are grouped by artifact type:

```text
manuscript_assets/figure2/
manuscript_assets/tables/
```

Figure-specific image assets belong under `manuscript_assets/figure<N>/`.
Reusable or supplementary TSV tables belong under `manuscript_assets/tables/`.

## Prepare Conservation Scores

Panel F uses gene-level mean phastCons100way scores over each gene's longest
protein-coding 3'UTR. If the conservation BigWig is not already available
locally, download it first:

```bash
uv run python scripts/download_phastcons100way.py
```

This downloads UCSC `hg38.phastCons100way.bw` to:

```text
data/resources/conservation/hg38.phastCons100way.bw
```

The BigWig is ignored by git because it is large.

Then compute the raw gene-level conservation table:

```bash
uv run python scripts/figure2_utr_conservation.py
```

By default this writes:

```text
manuscript_assets/tables/figure2F_utr_conservation_raw.tsv
```

The table contains `gene_id` and `utr3_mean_conservation`, plus supporting
transcript, scored-base, and 3'UTR-length columns. The main Figure 2 script
consumes this precomputed table so normal figure regeneration does not need to
re-read the multi-GB BigWig.

## Generate Figure 2

Run from the repository root:

```bash
uv run python scripts/figure2_coverage.py \
  --run-dir results/<results_dir> \
  --panel all
```

This writes the individual panels and the combined six-panel figure to:

```text
manuscript_assets/figure2/
```

It writes the panel TSVs and supplementary Figure 2 tables to:

```text
manuscript_assets/tables/
```

The Figure 2 script reads the ground-truth FDR and effect thresholds from the
completed benchmark run's config snapshot:

```text
<run-dir>/benchmark_config.yaml
```

Regenerate the benchmark run with the current pipeline if that file is missing.

The combined figure is:

```text
manuscript_assets/figure2/figure2_combined.png
manuscript_assets/figure2/figure2_combined.svg
```

The main table outputs are:

```text
manuscript_assets/tables/figure2A_experiment_coverage.tsv
manuscript_assets/tables/figure2B_gene_set_coverage.tsv
manuscript_assets/tables/figure2C_positive_coverage.tsv
manuscript_assets/tables/figure2D_background_coverage.tsv
manuscript_assets/tables/figure2E_utr_length.tsv
manuscript_assets/tables/figure2F_utr_conservation.tsv
manuscript_assets/tables/figure2F_utr_conservation_raw.tsv
manuscript_assets/tables/figure2_supplementary_coverage_table.tsv
manuscript_assets/tables/figure2_gene_set_overlap.tsv
manuscript_assets/tables/figure2_supplement_gene_set_overlap_mirna_coverage.tsv
```

The manuscript-focused supplement table
`figure2_supplement_gene_set_overlap_mirna_coverage.tsv` reports predictor
gene-set intersections and unions, plus the mean number of tested miRNAs scored
per covered gene for each predictor in the combination.

Panels:

- A: experiment coverage by predictor
- B: FGS/IGS gene-set coverage
- C: positive miRNA-gene pair coverage
- D: background miRNA-gene pair coverage
- E: longest protein-coding transcript 3'UTR length for IGS vs non-IGS genes
- F: mean 3'UTR conservation for IGS vs non-IGS genes

Panel E derives longest 3'UTR length from:

```text
data/resources/ensembl/Homo_sapiens.GRCh38.115.gtf.gz
```

For each gene, the script keeps one value: the longest 3'UTR among
protein-coding transcripts, after merging transcript-level 3'UTR intervals.

Panel F reads the precomputed gene-level conservation table:

```text
manuscript_assets/tables/figure2F_utr_conservation_raw.tsv
```

The table must contain:

```text
gene_id
utr3_mean_conservation
```

## Generate Performance Figures

Run from the repository root:

```bash
uv run python scripts/figure3_performance_naive.py \
  --run-dir results/<results_dir>
uv run python scripts/figure4_performance_ips.py \
  --run-dir results/<results_dir>
uv run python scripts/figure5_performance_fps.py \
  --run-dir results/<results_dir>
```

These figures evaluate miRNA-gene pair universes:

- Figure 3: algorithm-specific pairs. Each predictor is evaluated only on
  ground-truth pairs where that predictor supplied a score.
- Figure 4: Intersection Pair Set (IPS). All predictors are evaluated on the
  shared ground-truth pairs where every evaluated predictor supplied a score.
- Figure 5: Full Pair Set (FPS). All ground-truth pairs are evaluated. Missing
  predictor scores receive the zero-equivalent local rank.

Scores are converted to dataset-local normalized ranks where 0 is the weakest
scored pair and 1 is the strongest scored pair. Tied scores receive average
ranks. Figure 4 and Figure 5 include a deterministic random baseline generated
on the same evaluation universe as the compared predictors.

The combined figure panels show average precision, the top-100 median
perturbation-aware effect, AUROC, Spearman rho, an APS leaderboard, and a
Spearman R2 leaderboard. The top-100 median effect matches the top-prediction
CDF implementation used in the benchmark result plots: predictors are sorted by
score, the top N pairs are selected, and the median expected effect is reported.

These write separate panel assets and combined figure assets to:

```text
manuscript_assets/figure3/
manuscript_assets/figure4/
manuscript_assets/figure5/
```

It writes the supporting tables to:

```text
manuscript_assets/tables/figure3_algorithm_specific_per_experiment_metrics.tsv
manuscript_assets/tables/figure3_algorithm_specific_leaderboard.tsv
manuscript_assets/tables/figure3_algorithm_specific_spearman_r2_leaderboard.tsv
manuscript_assets/tables/figure4_ips_per_experiment_metrics.tsv
manuscript_assets/tables/figure4_ips_leaderboard.tsv
manuscript_assets/tables/figure4_ips_spearman_r2_leaderboard.tsv
manuscript_assets/tables/figure5_fps_per_experiment_metrics.tsv
manuscript_assets/tables/figure5_fps_leaderboard.tsv
manuscript_assets/tables/figure5_fps_spearman_r2_leaderboard.tsv
manuscript_assets/tables/figure5_fps_local_ranks.tsv
```

## Generate Figure 6 FPS Recovery

Run from the repository root:

```bash
uv run python scripts/figure6_fps_recovery.py \
  --run-dir results/<results_dir>
```

Figure 6 evaluates practical recovery under the Full Pair Set (FPS), using the
same rank-zero convention for missing predictor scores as Figure 5. It reports
positive-pair recovery by prediction budget, the best per-experiment precision
at a fixed recall target, and the corresponding false positives per true
positive.

This writes:

```text
manuscript_assets/figure6/figure6_fps_recovery_panel_a_recovery_budget.png
manuscript_assets/figure6/figure6_fps_recovery_panel_b_best_precision.png
manuscript_assets/figure6/figure6_fps_recovery_panel_c_validation_burden.png
manuscript_assets/figure6/figure6_fps_recovery_panel_d_legend.png
manuscript_assets/figure6/figure6_fps_recovery_combined.png
manuscript_assets/figure6/figure6_fps_recovery_combined.svg
manuscript_assets/tables/figure6_fps_recovery_by_budget.tsv
manuscript_assets/tables/figure6_fps_precision_at_recall.tsv
manuscript_assets/tables/figure6_fps_best_precision_at_recall.tsv
```

## Generate Supplementary Top-Effect CDF Figure

After running the benchmark with `evaluation.write_top_prediction_cdfs: true`,
combine the 30 per-dataset CDF diagnostic plots into supplementary figure
parts. The default layout uses 3 panels per row and 15 panels per part:

```bash
uv run python scripts/supplement_top_effect_cdfs.py \
  --run-dir results/<results_dir>
```

This writes:

```text
manuscript_assets/supplement/supplement_top_100_effect_cdfs_part1.png
manuscript_assets/supplement/supplement_top_100_effect_cdfs_part1.svg
manuscript_assets/supplement/supplement_top_100_effect_cdfs_part2.png
manuscript_assets/supplement/supplement_top_100_effect_cdfs_part2.svg
manuscript_assets/tables/supplement_top_100_effect_cdfs_manifest.tsv
```
