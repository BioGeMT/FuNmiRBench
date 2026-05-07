# Cheating

This directory contains the standardization pipeline for the strong demo-only `cheating` predictor.

## Files

- `pipeline.py`: CLI entrypoint for generating the standardized cheating predictor output.

## What The Pipeline Does

The pipeline:

1. Loads the experiment registry from `metadata/mirna_experiment_info.tsv`.
2. Restricts the input experiments to the shipped demo dataset IDs defined in `funmirbench.build_cheating_predictions`.
3. Reads the corresponding differential-expression tables.
4. Builds per `(miRNA, gene)` labels and directional signal from DE statistics.
5. Converts those signals into a bounded score in `[0, 1]`.
6. Writes the standardized prediction TSV.

This predictor is intentionally derived from benchmark-side DE information and is therefore for demo purposes only.

## Data Used

The pipeline uses:

- `metadata/mirna_experiment_info.tsv`
- the differential-expression tables referenced by `de_table_path` in that registry

Unlike `mock`, it restricts itself to the shipped demo dataset IDs defined in `funmirbench.build_cheating_predictions`:

- `GSE109725_OE_miR_204_5p`
- `GSE118315_KO_miR_124_3p`
- `GSE210778_OE_miR_375_3p`

## How Scores Are Generated

`cheating` is a strong demo-only predictor. It uses the selected DE tables to derive directional signal and positive labels from `logFC` and `FDR`, then combines those benchmark-side signals with controlled noise to generate bounded scores.

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
data/predictions/cheating/cheating_standardized.tsv
```

relative to the repository root.

## Run

From this directory:

```bash
python pipeline.py
```
