# GEO processing pipeline (placeholder)

This directory will contain the pipeline that converts raw GEO data
into differential expression (DE) tables compatible with FuNmiRBench.

Expected output format: TSV files with columns:

- gene_name
- logFC
- logCPM
- F
- PValue
- FDR

and filenames that match `metadata/datasets*.json` entries,
stored under `data/processed_GEO/`.