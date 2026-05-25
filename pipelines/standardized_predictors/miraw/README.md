# miRAW

This directory contains the standardization pipeline for miRAW gene-level predictions.

## Files

- `miraw_preprocessing.sh`: preprocessing script used to collapse raw miRAW site-level predictions to one best-scoring row per gene-miRNA pair.
- `pipeline.py`: CLI entrypoint for the pipeline.
- `utils.py`: shared helpers for logging, downloads, parsing, cleaning, mapping, and output construction.
- `miraw_pipeline.log`: log file written by the default run.

The raw miRAW predictions file is not downloaded by this pipeline. It must be obtained separately and placed locally.

The preprocessing script expects the raw miRAW predictions file at:

```text
data/all_ensg.txt.gz
```

and writes the best-per-pair preprocessed file to:

```text
data/best_per_pair.txt.gz
```

The standardization pipeline then reads the preprocessed miRAW predictions file. By default, or by passing `--predictions-file`, this should be:

```text
data/best_per_pair.txt.gz
```

The pipeline downloads and reuses annotation resources under:

```text
data/resources/
```

This includes:

- `mirbase/mature.fa` at `data/resources/mirbase/mature.fa`
- `biomart/hsapiens_ensembl_to_gene_name.tsv` at `data/resources/biomart/hsapiens_ensembl_to_gene_name.tsv`

If either cache file already exists, the pipeline reuses it. Otherwise, it downloads miRBase `mature.fa` version 22.1 and the BioMart Ensembl-to-gene-name mapping table, logging whether each resource was reused, downloaded, or failed.

The raw miRAW file is gzip-compressed and is treated as a line-oriented file containing one Python-dictionary-like prediction record per line.

Each record is expected to contain at least:

1. `GeneName`
2. `miRNA`
3. `Prediction`

miRAW produces site-level predictions. Therefore, the same gene-miRNA pair can appear many times with different `Prediction` scores and energy values. Before running the Python standardization pipeline, `miraw_preprocessing.sh` must be run to retain, for each gene-miRNA pair, the row with the highest `Prediction` score.

The preprocessing step extracts `GeneName`, `miRNA`, and `Prediction`, sorts by gene, miRNA, and descending prediction score, and keeps the first row for each gene-miRNA pair.

## What The Pipeline Does

The pipeline:

1. Uses the local preprocessed miRAW best-per-pair predictions file.
2. Downloads miRBase `mature.fa` version 22.1.
3. Downloads a BioMart TSV with headers and unique rows containing:
   - `Gene stable ID`
   - `Gene name`
4. Loads the preprocessed miRAW predictions file.
5. Parses the dictionary-like miRAW prediction rows.
6. Extracts:
   - `Ensembl_ID` from the `GeneName` field
   - `miRNA_Name` from the `miRNA` field
   - `Score` from the `Prediction` field
7. Drops rows with missing or invalid values in:
   - `Ensembl_ID`
   - `miRNA_Name`
   - `Score`
8. Deduplicates exact duplicate parsed rows.
9. Collapses rows to the best `Prediction` score per `(Ensembl_ID, miRNA_Name)` pair as an additional safety check.
10. Builds a miRNA mapping from human miRBase names (`hsa-*`) to `MIMAT` IDs.
11. Builds an Ensembl gene ID to `Gene_Name` mapping from BioMart.
12. Maps:
    - `miRNA_Name` to `miRNA_ID`
    - `Ensembl_ID` to `Gene_Name`
13. Uses the raw gene-name suffix as a fallback when an Ensembl ID cannot be mapped through BioMart.
14. Drops rows that still fail miRNA or gene-name mapping.
15. Converts `Prediction` to numeric `Score`.
16. Raises if the final `(Ensembl_ID, miRNA_ID)` pairs still have conflicting scores.
17. Drops exact duplicate final rows on:
    - `Ensembl_ID`
    - `miRNA_ID`
    - `Score`
18. Writes the standardized output table.

## Output Schema

The output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

`miRNA_Name` is copied from the raw miRAW `miRNA` field. `Score` is the numeric form of `Prediction`.

## Output Location

By default, the standardized file is written to:

```text
data/predictions/miraw/miraw_standardized.tsv
```

relative to the repository root.

## Run

From the repository root, first place the raw miRAW prediction file at:

```text
pipelines/standardized_predictors/miraw/data/all_ensg.txt.gz
```

Then run preprocessing:

```bash
cd pipelines/standardized_predictors/miraw
bash miraw_preprocessing.sh
```

This creates:

```text
data/best_per_pair.txt.gz
```

Then run the standardization pipeline from the repository root:

```bash
cd ../../..
uv run pipelines/standardized_predictors/miraw/pipeline.py
```

## CLI Arguments

Relative CLI paths are resolved from the repository root, so the current working directory does not change where inputs, resources, logs, or outputs are read or written.

```bash
uv run pipelines/standardized_predictors/miraw/pipeline.py \
  --predictions-file pipelines/standardized_predictors/miraw/data/best_per_pair.txt.gz \
  --resources-dir pipelines/standardized_predictors/miraw/data/resources \
  --output data/predictions/miraw/miraw_standardized.tsv \
  --log-file pipelines/standardized_predictors/miraw/miraw_pipeline.log \
  --log-level INFO
```

## Logging

Logging is written both to stdout and to the log file passed via `--log-file`.
