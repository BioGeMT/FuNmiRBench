# miRBind2 Gene-Level

This directory contains the standardization pipeline for miRBind2 gene-level
predictions.

## What The Pipeline Does

The pipeline:

1. Downloads or reuses the miRBind2 pretrained gene-level prediction CSV from
   Google Drive.
2. Reads the matching identifier-pickle download command from the miRBind2
   GitHub repo at
   `https://raw.githubusercontent.com/BioGeMT/miRBind_2.0/main/analysis/gene_level_model/download_data.sh`,
   then downloads that pickle into this pipeline's cache.
3. Downloads or reuses miRBase `mature.fa` release 22.1 to fill mature miRNA
   accessions.
4. Combines the prediction CSV and identifier pickle row by row.
5. Collapses duplicate `(Ensembl_ID, miRNA_Name)` pairs by keeping the lowest
   raw predicted fold-change value.
6. Writes the standardized FuNmiRBench predictor table.

## Output Schema

The output TSV contains:

- `Ensembl_ID`
- `Gene_Name`
- `miRNA_ID`
- `miRNA_Name`
- `Score`

`Score` is the raw miRBind2 predicted fold-change value. The predictor is
registered as `lower_is_stronger`, so FuNmiRBench converts it to the common
"higher is stronger" scale during benchmarking.

## Run

From this directory:

```bash
uv run pipeline.py
```

The final benchmark file is written to:

```text
data/predictions/mirbind_gene_level/mirbind_gene_level_standardized.tsv
```
