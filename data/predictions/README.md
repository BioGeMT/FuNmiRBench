# Precomputed prediction scores

This directory stores **precomputed scores from miRNA target prediction tools**.

Structure:

- One TSV file per target prediction algorithm, e.g.:

  - `data/predictions/targetscan.tsv`
  - `data/predictions/microt_cnn.tsv`

Each TSV should contain at least:

- `gene_name`
- `mirna_name`
- `score`

Metadata about these files will be stored in `metadata/predictions.json`
and accessed via the `funmirbench.predictions` module.
