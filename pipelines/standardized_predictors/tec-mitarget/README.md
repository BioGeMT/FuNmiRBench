# TEC-miTarget

This directory contains the standardization pipeline for TEC-miTarget gene-level predictions.

## Files

- `pipeline.py`: CLI entrypoint for the pipeline.
- `utils.py`: shared helpers for logging, downloads, cleaning, mapping, and output construction.
- `tec_mitarget_pipeline.log`: example log from a completed run.

The raw MRE-level prediction files used by this pipeline were shared by the respective authors via email and are stored as `test_split_*/predict.tsv` files under `data/TEC-miTarget-model-predictions`.

## What The Pipeline Does

The pipeline:

1. Downloads miRBase `mature.fa` version 22.1.
2. Downloads a headerless BioMart TSV with:
   - `Gene stable ID`
   - `Gene name`
   - `RefSeq mRNA ID`
3. Loads raw TEC-miTarget MRE-level `predict.tsv` files from `test_split_0` through `test_split_9`.
4. Drops rows with missing or invalid values in:
   - `query_ids`
   - `target_ids`
   - `predictions`
5. Deduplicates prediction rows on those same three columns.
6. Aggregates MRE-level predictions to transcript-level predictions with the max rule over each `(query_ids, target_ids)` pair.
7. Builds a miRNA mapping from human miRBase names (`hsa-*`) to `MIMAT` IDs.
8. Builds a RefSeq-to-gene mapping from BioMart after:
   - dropping invalid rows
   - dropping duplicate rows
   - dropping RefSeq IDs that map to more than one row
9. Maps:
   - `query_ids` to `miRNA_ID`
   - `target_ids` to `Ensembl_ID` and `Gene_Name`
10. Drops rows that fail either mapping.
11. Checks the final `(Ensembl_ID, miRNA_ID)` pairs, raising on conflicting scores and dropping exact duplicates.
12. Writes the standardized output table.

## Output Schema

The output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

`miRNA_Name` is copied from `query_ids`. `Score` is the numeric form of `predictions`.

## Output Location

By default, the standardized file is written to:

```text
data/predictions/tec-mitarget/tec_mitarget_standardized.tsv
```

relative to the repository root.

## Resource Cache

The pipeline downloads external resources only when the expected cache files are missing. By default, the cache files are:

```text
pipelines/standardized_predictors/tec-mitarget/data/resources/mirbase/mature.fa
pipelines/standardized_predictors/tec-mitarget/data/resources/biomart/hsapiens_refseq_to_ensembl.tsv
```

The log reports whether each resource is reused from cache or downloaded.

## Run

From the repository root:

```bash
uv run python pipelines/standardized_predictors/tec-mitarget/pipeline.py
```

## CLI Arguments

```bash
uv run python pipelines/standardized_predictors/tec-mitarget/pipeline.py \
  --predictions-root pipelines/standardized_predictors/tec-mitarget/data/TEC-miTarget-model-predictions \
  --resources-dir pipelines/standardized_predictors/tec-mitarget/data/resources \
  --output data/predictions/tec-mitarget/tec_mitarget_standardized.tsv \
  --log-file pipelines/standardized_predictors/tec-mitarget/tec_mitarget_pipeline.log \
  --log-level INFO
```

## Logging

Logging is written both to stdout and to the log file passed via `--log-file`. Main processing stages are logged as numbered steps, and row-count logs use a consistent `before -> after rows` format.
