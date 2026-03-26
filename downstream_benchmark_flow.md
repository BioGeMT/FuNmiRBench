# Downstream Benchmark Flow

This document describes only the downstream benchmark layer.
It assumes the upstream artifacts already exist.

Run entrypoint:

```bash
python -m funmirbench.cli.run_benchmark --config benchmark.example.yaml
```

Minimal runnable demo:

```bash
python -m funmirbench.cli.run_benchmark --config examples/dummy_benchmark/benchmark.yaml
```

## YAML Structure

```yaml
out_dir: results/run_example

experiments:
  dataset_ids:
    - "001"
  # or filters:
  # miRNA: ["hsa-miR-375-3p"]
  # cell_line: ["HUV-EC-C"]
  # perturbation: overexpression

predictors:
  tool_ids:
    - mock

evaluation:
  fdr_threshold: 0.05
  abs_logfc_threshold: 1.0
  predictor_top_fraction: 0.10
```

What each section means:

- `out_dir`
  - where the final `results/run_*` directory will be written
  - if omitted, the runner uses `results/run_benchmark`
- `experiments`
  - either exact `dataset_ids`
  - or filters: `miRNA`, `cell_line`, `perturbation`, `tissue`, `geo_accession`
- `predictors`
  - which `tool_ids` to load from `metadata/predictions.json`
  - optional `min_score` before joining
- `evaluation`
  - `fdr_threshold` and `abs_logfc_threshold` define GT positives
  - `predictor_top_fraction` defines the top fraction used in predictor-vs-predictor correlation

## Upstream Files We Consume

### 1. Experiment registry

```text
metadata/datasets.json
```

Important fields:

- `id`
- `miRNA`
- `cell_line`
- `perturbation`
- `geo_accession`
- `data_path`

### 2. Experiment GT table

```text
data/experiments/processed/*.tsv
```

Expected downstream shape:

- unnamed first column or explicit column containing gene IDs
- `logFC`
- `FDR`
- optional `PValue`

The join layer normalizes the gene IDs into an explicit downstream `gene_id` column.

### 3. Predictor registry

```text
metadata/predictions.json
```

Important fields:

- `tool_id`
- `canonical_tsv_path`

### 4. Canonical predictor scores

```text
data/predictions/<tool>/*_canonical.tsv
```

Expected columns:

- `mirna`
- `gene_id`
- `score`

## Input To Output Flowchart

```text
INPUTS
──────

  benchmark.yaml
    - out_dir
    - experiment selection
    - predictor selection
    - evaluation parameters

  metadata/datasets.json
    - dataset_id
    - miRNA
    - cell_line
    - perturbation
    - geo_accession
    - data_path

  data/experiments/processed/*.tsv
    - unnamed first column or explicit gene_id source
    - logFC
    - FDR
    - optional PValue

  metadata/predictions.json
    - tool_id
    - canonical_tsv_path

  data/predictions/<tool>/*_canonical.tsv
    - mirna
    - gene_id
    - score


FLOW
────

  benchmark.yaml
       │
       ▼
  run_benchmark.py
       │
       ├── set benchmark root = directory containing benchmark.yaml
       │
       ├── load metadata/datasets.json
       ├── load metadata/predictions.json
       │
       ├── select datasets
       │     - from dataset_ids
       │     - or from filters
       │
       ├── select tools
       │     - from tool_ids
       │
       └── for each selected dataset
             │
             ▼
      join_experiment_predictions.py
             │
             │  base table:
             │    experiment GT rows
             │    one row per GT gene
             │
             │  then for each tool:
             │    filter canonical scores to dataset miRNA
             │    left join on gene_id
             │    add column score_<tool_id>
             │
             ▼
      joined/<dataset_id>.tsv
             │
             │  columns:
             │  - dataset_id
             │  - mirna
             │  - gene_id
             │  - logFC
             │  - FDR
             │  - optional PValue
             │  - score_alpha
             │  - score_beta
             │  - ...
             │
             ▼
      plot_correlation.py
             │
             ├── per predictor:
             │     - score vs logFC scatter
             │     - PR curve / PR-AUC
             │     - ROC curve / AUROC
             │     - text report
             │
             ├── per dataset:
             │     - algorithms vs genes heatmap
             │     - predictor correlation heatmap
             │     - predictor correlation TSV
             │
             └── per run:
                   - APS table
                   - Spearman table
                   - AUROC table
                   - summary.json
                   - summary.txt


OUTPUTS
───────

  results/run_<name>/
  ├── joined/
  │   └── <dataset_id>.tsv
  │
  ├── plots/
  │   ├── <dataset>__<tool>_score_vs_logFC.png
  │   ├── <dataset>__<tool>_pr_curve.png
  │   ├── <dataset>__<tool>_roc_curve.png
  │   ├── <dataset>__algorithms_vs_genes_heatmap.png
  │   └── <dataset>__predictor_correlation_heatmap.png
  │
  ├── reports/
  │   ├── <dataset>__<tool>_evaluation_report.txt
  │   └── <dataset>__predictor_correlation.tsv
  │
  ├── tables/
  │   ├── aps_per_experiment.tsv
  │   ├── spearman_per_experiment.tsv
  │   └── auroc_per_experiment.tsv
  │
  ├── summary.json
  └── summary.txt
```

## What Is Working Now

- YAML config drives the run
- dataset selection supports exact IDs or filters
- one joined file is produced per dataset
- each tool becomes a `score_<tool_id>` column
- missing tool predictions remain `NA` in the joined table
- evaluation fills missing scores with `0.0` only for metric computation
- per-tool scatter, PR, and ROC outputs are produced
- dataset-level heatmaps are produced
- run-level APS, Spearman, and AUROC tables are produced

## What Is Still Missing

- a polished combined narrative benchmark report across all datasets and tools
- cross-run comparison management
- stricter formal schemas for the artifact files beyond the current fail-fast checks
