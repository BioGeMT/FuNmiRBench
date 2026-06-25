# Standardized Predictors

This directory contains pipelines that generate predictor outputs in a common standardized schema for downstream benchmarking.

## Setting up the environment

We provide a conda environment for running the standardized predictors.

```bash
conda env create -f environment.yml
conda activate standardized_predictors
```

Current predictor pipelines:

- `targetscan/`
- `tec-mitarget/`
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

The shared annotation schema uses Ensembl v115 (GRCh38) and miRBase release 22.1.

Not every pipeline currently populates every column, but they all write this schema.

## Pipelines

- `targetscan/`
- `tec-mitarget/`
- `microt_cnn/`
- `mirbind2/`
- `mirdb_mirtarget/`
- `miraw/`

See the README in each pipeline directory for pipeline-specific inputs, processing steps, and outputs.

## Logs and Zenodo Artifacts

Each predictor pipeline writes a TSV under `data/predictions/<tool_id>/` and a pipeline log under
its own directory. The default log files are:

- `targetscan/targetscan_pipeline.log`
- `tec-mitarget/tec_mitarget_pipeline.log`
- `microt_cnn/microt_cnn_pipeline.log`
- `mirbind2/mirbind2_pipeline.log`
- `mirdb_mirtarget/mirdb_mirtarget_pipeline.log`
- `miraw/miraw_pipeline.log`

These logs are generated artifacts and are ignored by Git. For publication reproducibility, include
the finalized logs with the standardized predictor/annotator TSVs in the Zenodo artifact. The
current Zenodo version contains the standardized TSVs; the next Zenodo version should also archive
these logs alongside any newly added experiments.
