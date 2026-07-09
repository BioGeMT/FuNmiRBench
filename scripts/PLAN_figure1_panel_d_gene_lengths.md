# Plan: Figure 1 panel D gene-length distribution

Goal: generate only the manuscript Panel D asset, a mirrored density plot comparing the 3'UTR length distribution of IGS genes versus non-IGS genes.

## Biological/statistical definition

- **FGS genes**: usable genes present in each dataset `joined.tsv` after the same manuscript/report filters used by the manuscript asset builder.
- **IGS genes**: FGS genes with non-missing scores from every selected predictor for that miRNA/dataset.
- **non-IGS genes**: FGS genes that are not scored by every selected predictor.
- **Length variable**: preferred x-axis is 3'UTR length in bp. This should come from a dedicated annotation table, not from the benchmark result directory itself.

Because predictor scoring is miRNA-specific, IGS membership should first be computed per dataset/miRNA. For the manuscript distribution, we should use one row per dataset-gene pair by default, then optionally support a unique-gene mode for sensitivity checks.

## Required inputs

1. Existing FuNmiRBench report directory, e.g.

   ```bash
   results/20260709_122904
   ```

   It must contain per-dataset `joined.tsv` files.

2. A gene-length annotation table, for example:

   ```text
   gene_id    utr3_length_bp
   ENSG...    1234
   ```

   Accepted aliases can be supported: `Ensembl_ID`, `ensembl_gene_id`, `gene`, `three_prime_utr_length`, `utr3_len`, `length_bp`.

3. Optional: if we want fully reproducible generation from Ensembl, add a helper script later to derive this table from a fixed Ensembl release GTF/BioMart export. For the figure builder itself, prefer a TSV input to keep the manuscript asset script fast and deterministic.

## Proposed implementation

Add a small, manuscript-only function set to `scripts/build_manuscript_figures_tables.py`:

1. `load_gene_length_table(path) -> pd.DataFrame`
   - Read TSV/CSV.
   - Normalize gene IDs by stripping Ensembl version suffixes.
   - Normalize the length column to `utr3_length_bp`.
   - Drop non-positive or missing lengths.
   - If multiple rows per gene exist, aggregate by max 3'UTR length by default. Add an option later for median/longest-canonical if needed.

2. `panel_d_gene_length_table(report_dir, gene_lengths, membership_mode='dataset_gene') -> pd.DataFrame`
   - Iterate over `joined.tsv` files.
   - Detect `score_<tool_id>` columns for selected predictors.
   - Classify each row as IGS if all selected score columns are non-missing.
   - Merge with `gene_lengths` on `gene_id`.
   - Return columns:
     - `dataset_id`
     - `gene_id`
     - `gene_set` with values `IGS genes` or `non-IGS genes`
     - `utr3_length_bp`
   - Write this table to `tables/figure1_panel_d_gene_lengths.tsv`.

3. `plot_figure1_panel_d_gene_lengths(table, figures_dir)`
   - Draw a mirrored KDE/ridge-style plot:
     - IGS genes above baseline.
     - non-IGS genes below baseline.
     - x-axis: `3'UTR Length (bp)`.
   - Save:
     - `figures/figure1_panel_d_gene_lengths.png`
     - `figures/figure1_panel_d_gene_lengths.svg`
   - Use only matplotlib/numpy/pandas, matching the current manuscript script dependencies.

4. CLI additions:

   ```bash
   uv run scripts/build_manuscript_figures_tables.py \
     --report-dir results/20260709_122904 \
     --out-dir manuscript_assets \
     --gene-lengths-tsv data/resources/gene_lengths/ensembl_v115_utr3_lengths.tsv
   ```

   If `--gene-lengths-tsv` is missing, skip Panel D with a clear warning. This keeps the current script usable with just benchmark outputs.

## Plot details

- Use the exact panel style from the manuscript mockup:
  - blue density above zero for IGS genes
  - red/orange density below zero for non-IGS genes
  - horizontal baseline at y=0
  - no y-axis ticks
  - compact legend at upper right
  - x-axis label: `3'UTR Length (bp)`
- Use a robust x-limit such as the 1st to 99th percentile, or fixed 0-4000 bp if we want identical manuscript panels across runs.
- Add `n=` labels on the left side for both classes.
- Avoid writing `SAMPLE` or placeholder labels in real outputs.

## Quality-control checks

The script should print or write a small QC summary:

- Number of joined files used.
- Number of dataset-gene rows classified as IGS and non-IGS.
- Number and percentage of rows with matched gene length.
- Number of genes dropped because length was unavailable.
- Median 3'UTR length for IGS and non-IGS.

Optional table output:

```text
gene_set    n_rows    n_unique_genes    median_utr3_length_bp    mean_utr3_length_bp
IGS genes
non-IGS genes
```

## Recommended next step

First add `--gene-lengths-tsv` support and generate Panel D from an uploaded or exported Ensembl v115 3'UTR length table. Then, after the figure looks right, optionally add a separate annotation-prep script that can create the gene-length table from a pinned Ensembl GTF/BioMart source.
