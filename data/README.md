# Data layout for FuNmiRBench

The **full differential expression (DE) tables** used in FuNmiRBench are **not stored in this repository** and are **not tracked by git**.

They should be placed under:

- `data/processed_GEO/` – processed DE tables (edgeR outputs, TSV)
- `data/raw_GEO/` – optional raw tables / intermediate files (if needed)

## Expected format of processed_GEO files

Each file in `data/processed_GEO/`:

- is a tab-separated file `.tsv`
- contains differential expression results for one experiment
- must have at least the following columns:

```text
gene_name    logFC    logCPM    F    PValue    FDR
