# microT-CNN

This directory contains the standardization pipeline for microT-CNN gene-level predictions.

## Files

- `pipeline.py`: CLI entrypoint for the pipeline.
- `utils.py`: helpers for logging, downloads, cleaning, mapping, and output construction.
- `microT_CNN_pipeline.log`: example log from a completed run (created when you run the pipeline).

## What The Pipeline Does

The pipeline:

1. Downloads miRBase `mature.fa` version 22.1.
2. Downloads the microT-CNN all-score predictions file (via Zenodo record by default).
3. Loads raw predictions and drops rows with missing/invalid values in the expected columns.
4. Deduplicates exact duplicate rows.
5. Maps miRNA human names (`hsa-*`) to `MIMAT` IDs using `mature.fa`.
6. Drops rows that fail miRNA name → MIMAT mapping.
7. Converts the prediction column to numeric `Score` and builds the final output table.
8. Writes the standardized TSV to the default location.

## Output Schema

The output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

`miRNA_Name` is copied from the raw miRNA name column. `Score` is the numeric form of the raw prediction value.

## Output Location

By default, the standardized file is written to:

```
data/predictions/microT_CNN/microT_CNN_standardized.tsv
```

relative to the repository root.

## Resource Cache

The pipeline downloads external resources only when the expected cache files are missing. By default, the cache files are:

```
pipelines/standardized_predictors/microT_CNN/data/resources/mirbase/mature.fa
pipelines/standardized_predictors/microT_CNN/data/microT_CNN_prediction_result_human_all_scores_gene_level.tsv.gz
```

The log notes whether each resource was reused from cache or downloaded.

## Run

From the repository root:

```bash
conda env create -f pipelines/standardized_predictors/environment.yml
conda activate standardized_predictors
python pipelines/standardized_predictors/microT_CNN/pipeline.py
```

Or using the repo's `uv` environment:

```bash
uv run pipelines/standardized_predictors/microT_CNN/pipeline.py
```

## CLI Arguments

```bash
python pipelines/standardized_predictors/microT_CNN/pipeline.py \
  --predictions-file pipelines/standardized_predictors/microT_CNN/data/microT_CNN_prediction_result_human_all_scores_gene_level.tsv.gz \
  --resources-dir pipelines/standardized_predictors/microT_CNN/data/resources \
  --output data/predictions/microT_CNN/microT_CNN_standardized.tsv \
  --log-file pipelines/standardized_predictors/microT_CNN/microT_CNN_pipeline.log \
  --log-level INFO
```

Relative CLI paths are resolved from the repository root.

## Logging

Logging is written to stdout and to the file passed via `--log-file`. Main processing stages are logged as numbered steps and row-count changes are logged in a `before -> after` format.
