# Standardized Predictors

This directory contains pipelines that generate predictor outputs in a common standardized schema for downstream benchmarking.

Current predictor pipelines:

- `targetscan/`
- `microt_cnn/`
- `mirbind2/`
- `mirdb_mirtarget/`
- `miraw/`

## Standardized Schema

The predictor outputs are written in a shared TSV format with the columns:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

The shared annotation schema uses Ensembl v115 (GRCh38) and miRBase release 22.1. Each standardized predictor table is expected to populate all five columns.

## Pipelines

- `targetscan/`
- `microt_cnn/`
- `mirbind2/`
- `mirdb_mirtarget/`
- `miraw/`

See the README in each pipeline directory for pipeline-specific inputs, processing steps, and outputs.
