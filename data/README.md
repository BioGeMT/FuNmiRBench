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
```
(The first column contains Ensembl gene IDs)

The filenames must match the paths specified in metadata/datasets.json, e.g.:
```json
{
  "id": "001",
  "data_path": "data/processed_GEO/GSE210778_edger_out_oe_hsa_miR_375_3p_oe.tsv",
  "miRNA": "hsa-miR-375-3p",
  "perturbation": "overexpression",
  ...
}
```
