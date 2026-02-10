# Metadata

FuNmiRBench separates **curated metadata** (what datasets exist and where their DE tables live, and what prediction tools exist and where their canonical score tables live)
from the **data files** themselves (which are not tracked by git).

## Dataset metadata flow

1. Curate `mirna_experiment_info.tsv` (source of truth)
2. Run `python scripts/build_experiments_index.py`
3. Commit the updated `datasets.json`

## Prediction tool registry flow

1. Curate `predictions_info.tsv` (source of truth)
2. Ensure the canonical TSV exists (e.g. `python scripts/build_predictions.py --tool mock`)
3. Run `python scripts/build_predictions_index.py`
4. Commit the updated `predictions.json`

## Files in this folder

## Dataset metadata

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
JSON list of dataset entries produced by `scripts/build_experiments_index.py`.

Each entry includes:
- `id`: FuNmiRBench dataset ID (e.g. `"001"`)
- `geo_accession`: GEO accession extracted from `gse_url` (e.g. `GSE210778`)
- `miRNA`, `miRNA_sequence`, `cell_line`, `tissue`, `perturbation`, `treatment`, `pubmed_id`, `gse_url`
- `data_path`: path to the processed DE TSV file (relative path string)

> Note: `data_path` is a pointer. The DE TSV files are **not stored in this repository**.

## Prediction tool registry

### `predictions_info.tsv` (input / curated)
Tab-separated table with one row per prediction tool.

This file is treated as **authoritative input** and should not be modified by the pipeline.
FuNmiRBench uses it to build a machine-readable tool registry (`predictions.json`).

Key columns:
- `tool_id`: short stable ID (e.g. `mock`)
- `official_name`: display name
- `organism`: e.g. `Homo sapiens`
- `score_type`, `score_direction`, `score_range`: semantics of the score
- `input_id_*_type`, `canonical_id_*_type`: identifier conventions
- `canonical_tsv_path`: repo-relative path to the tool’s canonical score TSV (like datasets’ `data_path`)

### `predictions.json` (generated registry)
JSON list of prediction tool entries produced by `scripts/build_predictions_index.py`.

Each entry includes the normalized fields from `predictions_info.tsv`, including `canonical_tsv_path`.

## Regenerating indexes

From the repository root:

```bash
# datasets.json
python scripts/build_experiments_index.py

# predictions.json
python scripts/build_predictions_index.py
```

## Why both TSV and JSON?

- The TSVs are **human-friendly** (easy to edit, review, and version-control).
- The JSON files are **machine-friendly** (stable schema, easy for Python code to load).

The TSVs should be treated as the **single source of truth**.
If a TSV and its JSON disagree, the TSV wins — regenerate the JSON.
