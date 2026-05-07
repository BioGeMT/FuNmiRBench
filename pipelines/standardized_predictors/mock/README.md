# Mock

This directory contains the standardization pipeline for the weak demo `mock` predictor.

## Files

- `pipeline.py`: CLI entrypoint for generating the standardized mock predictor output.

## What The Pipeline Does

The pipeline:

1. Loads the experiment registry from `metadata/mirna_experiment_info.tsv`.
2. Reads available differential-expression tables referenced by that registry.
3. Collects gene sets per miRNA from those tables.
4. Derives a weak signal from `logFC` and `FDR` where available.
5. Combines that signal with deterministic hash-based noise to generate reproducible mock scores.
6. Writes the standardized prediction TSV.

This predictor is intended as a weak demo baseline.

## Data Used

The pipeline uses:

- `metadata/mirna_experiment_info.tsv`
- the differential-expression tables referenced by `de_table_path` in that registry

It reads all available experiments it can resolve from the registry.

## How Scores Are Generated

`mock` is a weak demo predictor. It collects gene sets for each miRNA across the referenced DE tables, derives a weak signal from `logFC` and `FDR` where available, and combines that signal with deterministic hash-based noise to produce reproducible scores.

## Output Schema

The standardized output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

In the current implementation, only `Ensembl_ID`, `miRNA_Name`, and `Score` are populated. `Gene_Name` and `miRNA_ID` are written as empty fields.

## Output Location

The standardized file is written to:

```text
data/predictions/mock/mock_standardized.tsv
```

relative to the repository root.

## Run

From this directory:

```bash
python pipeline.py
```
