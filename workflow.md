
# FuNmiRBench Architecture Overview

```
                     FuNmiRBench Architecture
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                         UPSTREAM                            │
│                (experiment preparation)                     │
│                                                             │
│  Raw RNA-seq experiments       or Already proceesed ones    │
│          │                                                  │
│          ▼                                                  │
│   RNA-seq processing pipeline                               │
│   (alignment, quantification, DE analysis)                  │
│          │                                                  │
│          ▼                                                  │
│   Differential expression tables                            │
│   gene | logFC | FDR                                        │
│          │                                                  │
│          ▼                                                  │
│   metadata/mirna_experiment_info.tsv                        │
│          │                                                  │
│          ▼                                                  │
│   metadata/datasets.json                                    │
│                                                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ dataset index + DE tables
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                        DOWNSTREAM                           │
│                   (benchmark engine)                        │
│                                                             │
│        Dataset selection                                    │
│        (miRNA / cell line / perturbation)                   │
│                        │                                    │
│                        ▼                                    │
│        Prediction algorithms                                │
│        (mock / TargetScan / mirDB / etc.)                 │
│                        │                                    │
│                        ▼                                    │
│        Join predictions with experiments                    │
│                                                             │
│        gene | score | logFC | FDR                           │
│                                                             │
│                        ▼                                    │
│        Evaluation metrics (?)                                   │
│        • PR curves                                          │
│        • ROC curves                                         │
│        • enrichment curves                                  │
│        • score vs logFC                                     │
│        • CDF plots                                          │
│                                                             │
│                        ▼                                    │
│        Benchmark outputs (?)                                    │
│                                                             │
│        results/run_*/                                       │
│        ├── plots/                                           │
│        ├── reports/                                         │
│        ├── joined tables                                    │
│        └── comparison summaries                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# Simplified Concept

FuNmiRBench compares **miRNA target prediction algorithms** with **real biological experiments**.

```
Algorithm predictions
        vs
Real gene expression changes
```

The benchmark checks whether genes predicted as targets actually change expression in experiments.

---

# Upstream Pipeline

The upstream pipeline prepares **ground truth experimental data**.

Should prepare predictors also?

---

# Experiment Metadata

File:

```
metadata/mirna_experiment_info.tsv
```

Example entry:

```
mirna_name       hsa-miR-21-5p
cell_line        HeLa
experiment_type  OE
gse_url          GSE12345
de_table_path    exp1.tsv
```

This file links:

- biological context (miRNA, cell line, experiment)
- differential expression data file

---

# Dataset Index

Script:

```
build_experiments_index.py
```

Converts the TSV metadata into:

```
metadata/datasets.json
```

Example entry:

```json
{
  "id": "EXP001",
  "miRNA": "hsa-miR-21-5p",
  "cell_line": "HeLa",
  "perturbation": "overexpression",
  "data_path": "data/experiments/processed/exp1.tsv"
}
```

This JSON file allows the benchmark engine to load experiments programmatically.

---

# Downstream Benchmark

The downstream pipeline evaluates prediction algorithms.

Example predictors:

```
mock
TargetScan
mirDB
```

Predictions look like:

```
gene  mirna          score
TP53  hsa-miR-21-5p  0.89
MYC   hsa-miR-13-3p  0.12
...
```

---

# Joining Predictions with Experiments

Predictions are merged with experimental results:

```
gene      score      logFC     FDR
TP53      0.89       -1.3      0.001
MYC       0.12        0.7      0.03
...
```

This is the **core evaluation table**.

---

# Evaluation Metrics - To discuss

### Scatter plots

```
score vs logFC
score vs -log10(FDR)
```

### Classification metrics

```
PR curve
ROC curve
```

### Distribution analysis

```
CDF plots
boxplots
violin plots
```

### Enrichment analysis

```
GSEA-style enrichment curves
```

---

# Final Output Structure - To discuss

Each benchmark run produces a structured results directory:

```
results/
└── run_<timestamp>/
    ├── run_summary.tsv
    ├── datasets/
    │   ├── EXP001/
    │   │   └── predictors/
    │   │       ├── mock/
    │   │       │   ├── joined/
    │   │       │   ├── plots/
    │   │       │   └── reports/
    │   │       └── targetscan/
    │   │           ├── joined/
    │   │           ├── plots/
    │   │           └── reports/
    └── comparisons/
        ├── across_predictors/
        └── across_datasets/
```

---

# Summary

```
RNA-seq experiments
        │
        ▼
Differential expression (logFC, FDR)
        │
        ▼
FuNmiRBench benchmark
        │
        ▼
Prediction algorithm evaluation
```
