# TargetScan

This directory contains the standardization pipeline for TargetScan v8 gene-level predictions.

## Files

- `pipeline.py`: CLI entrypoint for the pipeline.
- `utils.py`: shared helpers for logging, downloads, parsing, QC, mapping, and output construction.
- `targetscan_pipeline.log`: example log from a completed run.
- `data/`: cached TargetScan, Ensembl, and miRBase resources used by the pipeline.

## What The Pipeline Does

The pipeline:

1. Downloads or reuses the TargetScan v8 input files:
   - `Summary_Counts.all_predictions.txt`
   - `miR_Family_Info.txt`
   - `Gene_info.txt`
2. Builds a representative-transcript index from `Gene_info.txt` for human rows only (`Species ID = 9606`).
3. Keeps every transcript with `Representative transcript? == 1`.
4. Logs whether any gene has more than one representative transcript row.
5. Downloads or reuses the Ensembl v115 GTF for GRCh38.
6. Builds cached Ensembl lookup tables for:
   - transcript stable ID to gene stable ID
   - gene stable ID to gene name
7. Runs QC comparing TargetScan representative transcripts against Ensembl transcript and gene mappings.
8. Downloads or reuses miRBase `mature.fa` version 22.1.
9. Builds a human miRNA mapping from TargetScan `MiRBase ID` to:
   - `miRNA_ID` as miRBase accession
   - `miRNA_Name` as the miRBase mature name
10. Reads `Summary_Counts.all_predictions.txt` and keeps only human rows.
11. Drops rows where:
   - the transcript is not in the representative transcript set
   - the transcript does not map to an Ensembl gene
   - the TargetScan gene ID for that transcript disagrees with the Ensembl gene ID recovered from the transcript
   - the representative miRNA cannot be mapped through the miRNA annotation table
   - the score is null
12. Preserves the raw TargetScan `Cumulative weighted context++ score` as `Score`.
13. Ensures the final output has one row per `(Ensembl_ID, miRNA_ID)` pair.
14. Writes the standardized output table.

## Output Schema

The output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

`Score` is the raw numeric form of TargetScan `Cumulative weighted context++ score`.

## Run

From this directory:

```bash
uv run pipeline.py
```

## CLI Arguments

```bash
uv run pipeline.py --log-level INFO
```

Supported log levels:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

## Logging

Logging is written both to stdout and to `targetscan_pipeline.log`.
