# Data layout for FuNmiRBench

The **full differential expression (DE) tables** used in FuNmiRBench are **not stored in this repository** and are **not tracked by git**.

They should be placed under:

- `data/experiments/processed/` – processed DE tables (edgeR outputs, TSV)
- `data/raw_GEO/` – optional raw tables / intermediate files (if needed)

## Getting the published benchmark corpus (Zenodo)

FuNmiRBench publishes the processed DE tables as a Zenodo record (files may be restricted to link requests).

1. Obtain an access token (from the link request / Zenodo UI).
2. Download all files into `data/experiments/processed/`:

```bash
# either provide token explicitly...
python -m funmirbench.cli.download_zenodo_corpus --token "<TOKEN>"

# ...or via environment variable
export ZENODO_TOKEN="<TOKEN>"
python -m funmirbench.cli.download_zenodo_corpus
```

The record DOI is: `10.5281/zenodo.17585186`.

## Expected format of processed experiment tables

Each file in `data/experiments/processed/`:

- is a tab-separated file `.tsv`
- contains differential expression results for one experiment

edgeR outputs do **not** include a gene symbol column by default. FuNmiRBench only requires a stable gene identifier column,
typically the first column (often Ensembl gene IDs). If available, the column name may be `gene_name` or `gene_id`.

Common columns include:

```text
gene_name    logFC    logCPM    F    PValue    FDR
```

The filenames must match the paths specified in `metadata/datasets.json`, e.g.:

```json
{
  "id": "001",
  "data_path": "data/experiments/processed/GSE210778_edger_out_oe_hsa_miR_375_3p_oe.tsv",
  "miRNA": "hsa-miR-375-3p",
  "perturbation": "overexpression"
}
```
