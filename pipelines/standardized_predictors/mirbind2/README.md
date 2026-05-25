# miRBind2

This directory contains the standardization pipeline for miRBind2 human 3UTR predictions.

## Files

- `pipeline.py`: CLI entrypoint for the pipeline.
- `utils.py`: helpers for logging, cleaning, MIMAT annotation, and output construction.
- `mirbind2_pipeline.log`: log file written by the default run.

## Inputs

The raw input file is not tracked in Git because it is large. Before running the
pipeline, copy the raw miRBind2 prediction table into this directory with this
exact filename:

```text
unified_human_mirbase_mane_select_3utrs_selected_18_mirnas_mirbind2_predictions.tsv
```

Its header is:

```text
noTranscript_ID	Gene_ID	Gene_Symbol	miRNA_Name	miRNA_Sequence	UTR_sequence	miRBind2_3UTR_prediction
```

The pipeline validates these raw columns:

- `noTranscript_ID`
- `Gene_ID`
- `Gene_Symbol`
- `miRNA_Name`
- `miRNA_Sequence`
- `UTR_sequence`
- `miRBind2_3UTR_prediction`

The pipeline uses the existing shared miRBase resource:

```text
data/resources/mirbase/mature.fa
```

This is miRBase release 22.1. The raw miRBind2 table already contains Ensembl gene IDs and gene symbols, so no BioMart remapping is performed.

## What The Pipeline Does

The pipeline:

1. Loads the raw miRBind2 TSV.
2. Validates the raw miRBind2 columns listed above.
3. Drops rows with invalid values in `Gene_ID`, `miRNA_Name`, or `miRBind2_3UTR_prediction`.
4. Deduplicates exact raw rows on those same columns.
5. Builds a human miRNA name to `MIMAT` ID mapping from miRBase 22.1 `mature.fa`.
6. Annotates `miRNA_Name` to `miRNA_ID`.
7. Drops rows whose miRNA names cannot be mapped to `MIMAT` IDs.
8. Drops raw transcript and sequence payload columns:
   - `noTranscript_ID`
   - `miRNA_Sequence`
   - `UTR_sequence`
9. Converts `miRBind2_3UTR_prediction` to numeric `Score`.
10. Writes the standardized output table.

## Output Schema

The output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

`Score` is the numeric form of `miRBind2_3UTR_prediction`.
Missing raw `Gene_Symbol` values are retained as blank `Gene_Name` values.

The score is a signed raw miRBind2-3UTR repression prediction, not a probability.
Lower, more negative scores are treated as stronger predicted repression.

## Output Location

By default, the standardized file is written to:

```text
data/predictions/mirbind2/mirbind2_standardized.tsv
```

relative to the repository root.

## Run

From the repository root:

```bash
conda run -n standardized_predictors \
  python pipelines/standardized_predictors/mirbind2/pipeline.py
```

## CLI Arguments

Relative CLI paths are resolved from the repository root.

```bash
conda run -n standardized_predictors \
  python pipelines/standardized_predictors/mirbind2/pipeline.py \
  --predictions-file pipelines/standardized_predictors/mirbind2/unified_human_mirbase_mane_select_3utrs_selected_18_mirnas_mirbind2_predictions.tsv \
  --mirbase-mature data/resources/mirbase/mature.fa \
  --output data/predictions/mirbind2/mirbind2_standardized.tsv \
  --log-file pipelines/standardized_predictors/mirbind2/mirbind2_pipeline.log \
  --log-level INFO
```

## Logging

Logging is written both to stdout and to the log file passed via `--log-file`.
