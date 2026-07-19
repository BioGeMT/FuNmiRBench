# miRAW

This directory contains the standardization pipeline for miRAW gene-level predictions.

## Files

- `miraw_preprocessing.sh`: legacy preprocessing script for local dictionary-like miRAW dumps.
- `preprocess_miraw.py`: legacy Python reference implementation for the best-per-pair preprocessing logic.
- `pipeline.py`: CLI entrypoint for the pipeline.
- `utils.py`: shared helpers for logging, downloads, parsing, cleaning, mapping, and output construction.
- `miraw_pipeline.log`: log file written by the default run.

The pipeline downloads the raw miRAW site-level prediction TSV from Figshare DOI `10.6084/m9.figshare.32982218` and caches it at:

```text
data/helios_summary.tsv.gz
```

The legacy preprocessing scripts are retained only for older local dictionary-like miRAW dumps. They are not part of the default Figshare-based run.

If you need to use the legacy preprocessing path, place the raw dictionary-like miRAW prediction file at:

```text
data/all_ensgs.txt.gz
```

and run the legacy preprocessing script to write:

```text
data/best_per_pair.tsv.gz
```

The standardization pipeline can still parse legacy preprocessed files internally, but the CLI uses the Figshare source only.

The default Figshare file is a TSV with site-level miRAW predictions. It contains the columns:

```text
Target_ENSG, GeneName, miRNA, Prediction
```

The pipeline downloads and reuses annotation resources under:

```text
data/resources/
```

This includes:

- `mirbase/mature.fa` at `data/resources/mirbase/mature.fa`
- `ensembl/Homo_sapiens.GRCh38.115.gtf.gz` at `data/resources/ensembl/Homo_sapiens.GRCh38.115.gtf.gz`

If either cache file already exists, the pipeline reuses it. Otherwise, it downloads miRBase `mature.fa` version 22.1 and the Ensembl v115 GTF, logging whether each resource was reused, downloaded, or failed.

miRAW produces site-level predictions. Therefore, the same Ensembl gene-miRNA pair can appear many times with different `Prediction` scores. The Python standardization pipeline collapses these rows internally and keeps the highest score for each `(Ensembl_ID, miRNA_Name)` pair.

## What The Pipeline Does

The pipeline:

1. Downloads or reuses the Figshare miRAW site-level TSV.
2. Downloads or reuses miRBase `mature.fa` version 22.1.
3. Downloads the Ensembl v115 Homo sapiens GTF used for gene-name annotation.
4. Loads and validates raw miRAW prediction rows.
5. Parses the Figshare TSV.
6. Extracts:
   - `Ensembl_ID` from `Target_ENSG`/`GeneName`
   - `miRNA_Name` from the `miRNA` field
   - `Score` from the `Prediction` field
7. Drops rows with missing or invalid values in:
   - `Ensembl_ID`
   - `miRNA_Name`
   - `Score`
8. Deduplicates exact duplicate parsed rows.
9. Collapses rows to the best `Prediction` score per `(Ensembl_ID, miRNA_Name)` pair as an additional safety check.
10. Builds a miRNA mapping from human miRBase names (`hsa-*`) to `MIMAT` IDs.
11. Builds an Ensembl gene ID to `Gene_Name` mapping from Ensembl v115 `gene` features.
12. Maps:
    - `miRNA_Name` to `miRNA_ID`
    - `Ensembl_ID` to `Gene_Name`
13. Drops genes absent from Ensembl release 115.
14. Converts `Prediction` to numeric `Score`.
15. Keeps one highest-scoring final row per:
    - `Ensembl_ID`
    - `miRNA_ID`
16. Writes the standardized output table.

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

From the repository root, run the standardization pipeline:

```bash
uv run pipelines/standardized_predictors/miraw/pipeline.py
```

## CLI Arguments

Relative CLI paths are resolved from the repository root, so the current working directory does not change where inputs, resources, logs, or outputs are read or written.

```bash
uv run pipelines/standardized_predictors/miraw/pipeline.py \
  --resources-dir pipelines/standardized_predictors/miraw/data/resources \
  --output data/predictions/miraw/miraw_standardized.tsv \
  --log-file pipelines/standardized_predictors/miraw/miraw_pipeline.log \
  --log-level INFO
```

## Logging

Logging is written both to stdout and to the log file passed via `--log-file`.
