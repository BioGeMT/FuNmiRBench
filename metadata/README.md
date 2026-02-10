# Metadata

FuNmiRBench separates **curated metadata** (what datasets exist and where their DE tables live)
from the **data files** themselves (which are not tracked by git).

The intended flow is:

1. Curate `mirna_experiment_info.tsv` (source of truth)
2. Run `python scripts/build_index.py`
3. Commit the updated `datasets.json`

## Files in this folder

### `mirna_experiment_info.tsv` (input / curated)
Tab-separated table with one row per experiment/dataset.

This file is treated as **authoritative input** and should not be modified by the pipeline.
FuNmiRBench uses it to build a machine-readable index (`datasets.json`).

Key columns:
- `mirna_name`: miRNA identifier (e.g. `hsa-miR-375-3p`)
- `mirna_sequence`: mature miRNA sequence
- `article_pubmed_id`: PubMed URL or ID
- `tested_cell_line`: cell line (optional)
- `treatment`: free-text description (optional)
- `tissue`: tissue (optional)
- `experiment_type`: `OE`, `KD`, or `KO`
- `gse_url`: GEO record URL (source information)
- `de_table_path`: filename of the processed DE table (TSV) for this experiment

### `datasets.json` (generated index)
JSON list of dataset entries produced by `scripts/build_index.py`.

Each entry includes:
- `id`: FuNmiRBench dataset ID (e.g. `"001"`)
- `geo_accession`: GEO accession extracted from `gse_url` (e.g. `GSE210778`)
- `miRNA`, `miRNA_sequence`, `cell_line`, `tissue`, `perturbation`, `treatment`, `pubmed_id`, `gse_url`
- `data_path`: path to the processed DE TSV file (relative path string)

> Note: `data_path` is a pointer. The DE TSV files are **not stored in this repository**.

## Regenerating `datasets.json`

From the repository root:

```bash
python scripts/build_index.py
```

## Why both TSV and JSON?

- The TSV is **human-friendly** (easy to edit, review, and version-control).
- The JSON is **machine-friendly** (stable schema, easy for Python code to load).

The TSV should be treated as the **single source of truth**.
If the TSV and JSON disagree, the TSV wins — regenerate the JSON.
