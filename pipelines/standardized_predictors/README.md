# Standardized Predictors

This directory contains pipelines that generate predictor outputs in a common standardized schema for downstream benchmarking.

## Setting up the environment

We provide a conda environment for running the standardized predictors.

```bash
conda env create -f environment.yml
conda activate standardized_predictors
```

Current predictor pipelines:

- `mock/`
- `cheating/`
- `targetscan/`
- `tec-mitarget/`

## Standardized Schema

The predictor outputs are written in a shared TSV format with the columns:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

The shared annotation schema uses Ensembl v115 (GRCh38) and miRBase release 22.1.

Not every demo pipeline currently populates every column, but they all write this schema.

## Pipelines

- `targetscan/`
- `tec-mitarget/`
- `mock/`
- `cheating/`

See the README in each pipeline directory for pipeline-specific inputs, processing steps, and outputs.
