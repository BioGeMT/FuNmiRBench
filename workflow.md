# FuNmiRBench Workflow

FuNmiRBench evaluates miRNA perturbation experiments listed in a single registry file:

**metadata/mirna_experiment_info.tsv**

This file is the **single source of truth** for experiments used in benchmarking.

If an experiment appears in this TSV, FuNmiRBench assumes:

* A processed differential expression table already exists
* The table location is specified in `de_table_path`

---

# Workflow Overview

The benchmark workflow consists of five logical stages:

1. Experiment preparation (upstream)
2. Build experiment index
3. Validate experiments
4. Join with predictors (upstream standardization)
5. Evaluate predictors

---

# 1. Experiment Preparation (Upstream)

Experiments must be prepared **before running the benchmark**.

Possible sources:

### A. Zenodo benchmark corpus

Experiments are downloaded and stored locally.

### B. User processed experiments

A user may already have processed differential expression tables.

### C. Raw RNA-seq data

Users may process raw RNA-seq data using the external pipeline. (pipelines/geo)


This upstream pipeline has its **own Conda environment**. (including everything needed for standardize predictors also)

Once a processed table exists, the user adds a row to:

metadata/mirna_experiment_info.tsv

Should we give the user the ability to include his own predictors?
---

# 2. Experiment Registry

The registry file contains metadata for each experiment.

Example fields:

* mirna_name
* mirna_sequence
* article_pubmed_id
* tested_cell_line
* treatment
* tissue
* experiment_type
* gse_url
* de_table_path

Important rule:

Every row must point to a **valid processed DE table** via `de_table_path`.

---

# 3. Build Experiment Index

The registry is converted into a structured metadata file:

datasets.json

Command:

build_experiments_index

Purpose:

* assign dataset IDs
* store experiment locations

---

# 4. Validate Experiments

The validation step ensures all experiments are usable.

Command:

validate_experiments

Checks include:

* files exist
* files are readable
* gene ID column present
* DE table structure valid

If validation fails, the benchmark should **not continue**.

---

# 5. Run Predictors

Predictors generate target scores for each miRNA.

Example predictor tools:

* mock (testing)
* TargetScan
* microT-CNN
* others

Predictions are stored under:

data/predictions/<tool>/

Example:

data/predictions/mock/mock_canonical.tsv

Command example:

build_predictions --tool mock

---

# 6. Join Predictions with Experiments

Predicted targets are merged with differential expression data.

Command:

join_experiment_predictions

Output:

data/joined/<dataset>_<tool>.tsv

Example:

data/joined/008_mock.tsv

---

# 7. Evaluation

Evaluation measures the relationship between predicted scores and gene expression changes.

Example metrics:

* Pearson correlation
* Spearman correlation

Command:

plot_correlation

Outputs:

data/plots/<dataset>_<tool>_score_vs_logFC.png

---

# Key Design Decisions

### Single Source of Truth

mirna_experiment_info.tsv defines the benchmark dataset.

### Separation of Responsibilities

Experiment preparation/ Predictors → upstream pipelines
Benchmark evaluation → FuNmiRBench


