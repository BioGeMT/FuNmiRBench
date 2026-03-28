# Data Layout

`data/` holds benchmark inputs, not benchmark outputs.

Current policy:

- `data/experiments/processed/`: tracked experiment DE tables used by the benchmark
- `data/predictions/`: local generated predictor TSVs, not tracked by git
- benchmark outputs go to `results/`, not `data/`

The benchmark reads paths from:

- `metadata/mirna_experiment_info.tsv` via `de_table_path`
- `metadata/predictions_info.tsv` via `canonical_tsv_path`

Expected layout:

```text
data/
  experiments/processed/
  predictions/
```

DE tables must contain gene identifiers plus `logFC` and `FDR`. `PValue` is optional.

Canonical prediction tables must contain:

```text
mirna    gene_id    score
```

Demo predictor files are generated locally with:

```bash
uv run pipelines/standardized_predictors/mock/pipeline.py
uv run pipelines/standardized_predictors/cheating/pipeline.py
```

This writes:

```text
data/predictions/mock/mock_canonical.tsv
data/predictions/cheating/cheating_canonical.tsv
```

Older paths like `data/joined/` and `data/plots/` are obsolete. Current runs write joined tables,
plots, reports, and summaries under `results/`.
