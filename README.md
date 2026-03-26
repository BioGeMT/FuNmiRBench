# FuNmiRBench

Benchmarking framework for **functional microRNA target prediction**.

FuNmiRBench provides:

- Standardized metadata for >50 functional miRNA perturbation RNA-seq datasets (currently sourced from GEO)
- A unified Python API to:
  - list and filter datasets by miRNA, cell line, perturbation type, tissue, and GEO accession
  - load differential expression tables (edgeR outputs) into pandas
  - summarize the available datasets
- A downstream benchmark workflow for joining canonical predictor scores to experiment GT tables and producing evaluation outputs

---

## 🔧 Installation

FuNmiRBench supports two installation modes:

### Option 1 (recommended): reproducible conda environment (pinned versions)

Clone the repo:

```bash
git clone git@github.com:BioGeMT/FuNmiRBench.git
cd FuNmiRBench
```
Create and activate the pinned conda environment:
```bash
conda env create -f environment.yml
conda activate funmirbench
```
This installs core dependencies via conda (with fixed versions) and installs the
package in editable mode (pip install -e .).

### Option 2: manual development setup (pip-managed deps)
```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda create -n funmirbench python=3.12
conda activate funmirbench

pip install -e .
```
## Current Scope

FuNmiRBench currently does three practical things:

1. manages experiment and predictor registries
2. validates and loads processed experiment / predictor artifacts
3. runs a downstream benchmark from those artifacts

Important boundary:

- this repo does not currently compute DE tables from raw RNA-seq
- experiment DE tables are imported or downloaded as processed TSVs
- predictor TSVs are generated here only for the currently implemented tools, which is `mock` right now

## 📂 Project structure

```
FuNmiRBench/
├── data/
│   ├── README.md                 # expected runtime data layout
│   ├── experiments/              # processed GT tables
│   └── predictions/              # canonical predictor TSVs
│
├── metadata/
│   ├── mirna_experiment_info.tsv # curated experiment metadata
│   ├── datasets.json             # generated experiment registry
│   ├── predictions_info.tsv      # curated predictor metadata
│   └── predictions.json          # generated predictor registry
│
├── pipelines/
│   └── geo/
│       └── README.md             # notes on the external GEO -> DE pipeline
│
├── src/
│   └── funmirbench/
│       ├── __init__.py
│       ├── datasets.py           # dataset metadata and loading API
│       ├── de_table_validation.py
│       ├── utils/                # shared path / TSV / gene-id helpers
│       └── cli/                  # operational script entrypoints
│           ├── import_experiments.py
│           ├── build_experiments_index.py
│           ├── validate_experiments.py
│           ├── build_predictions.py
│           ├── build_predictions_index.py
│           ├── join_experiment_predictions.py
│           ├── plot_correlation.py
│           ├── run_benchmark.py
│           └── smoke_run.py
│
└── tests/
    └── test_*.py
```

FuNmiRBench is currently script-first.
The main reusable library pieces are `src/funmirbench/datasets.py` and `src/funmirbench/utils/`.
Most pipeline behavior lives in the CLI scripts under `src/funmirbench/cli/`.

## Main Commands

Experiment-side:

```bash
# Import processed DE tables
python -m funmirbench.cli.import_experiments --from-dir /path/to/processed_tables

# Build experiment registry
python -m funmirbench.cli.build_experiments_index

# Validate referenced experiment tables
python -m funmirbench.cli.validate_experiments
```

Predictor-side:

```bash
# Build canonical predictor TSVs
python -m funmirbench.cli.build_predictions --tool mock

# Build predictor registry
python -m funmirbench.cli.build_predictions_index
```

Downstream benchmark:

```bash
python -m funmirbench.cli.run_benchmark --config benchmark.example.yaml
```

Minimal runnable example:

```bash
python -m funmirbench.cli.run_benchmark \
  --config examples/dummy_benchmark/benchmark.yaml
```

---

## 📊 Using the dataset API

Basic loading:

```python
from funmirbench import load_dataset, load_all_datasets

# Load a single dataset by ID (e.g. "001")
df = load_dataset("001")

# Load all datasets
df_all = load_all_datasets()

# Filter by perturbation
df_oe = load_all_datasets(perturbation="overexpression")
df_kd = load_all_datasets(perturbation="knockdown")
```

Metadata exploration:

```python
from funmirbench import datasets

datasets.list_mirnas()
datasets.list_cell_lines()
datasets.list_perturbations()
datasets.summarize_cell_lines()
datasets.summarize_mirnas()

summaries = datasets.summarize_datasets()
```

## 🚀 Workflow Overview

The artifact flow is:

```text
metadata/mirna_experiment_info.tsv
  + processed experiment TSVs
  -> metadata/datasets.json

metadata/predictions_info.tsv
  + canonical predictor TSVs
  -> metadata/predictions.json

datasets.json
  + experiment TSVs
  + predictions.json
  + canonical predictor TSVs
  -> run_benchmark.py
  -> results/run_*/
```

For the full downstream artifact contract and output layout, see [downstream_benchmark_flow.md](./downstream_benchmark_flow.md).

### 1. Import experiment tables

Download the published benchmark experiments (processed DE tables) from Zenodo
into `data/experiments/processed/`:

```bash
python -m funmirbench.cli.import_experiments --token "<TOKEN>"
```

- The token is provided by Zenodo for restricted records  
- Alternatively, set `ZENODO_TOKEN` as an environment variable  
- This step only downloads or copies processed DE tables.
- It does not compute DE results from raw RNA-seq inside this repo.

If you already have processed DE tables locally, import them instead of using Zenodo:

```bash
python -m funmirbench.cli.import_experiments --from-dir /path/to/processed_tables
```

- Copies top-level `.tsv` files into `data/experiments/processed/` (or `--out-dir`)
- Validates that each TSV is readable and that gene identifiers can be detected
- Use `--overwrite` to replace existing files

To combine Zenodo and local tables, run Zenodo import first, then run `--from-dir`
to add your extra local tables. Use `--overwrite` only when you explicitly want local
files to replace same-named files already present in the output folder.

---

### 2. Build the experiment index

Generate the machine-readable dataset registry from curated metadata:

```bash
python -m funmirbench.cli.build_experiments_index
```

This produces:

- `metadata/datasets.json`

which links experiment metadata to local DE table paths.

---

### 3. Validate local experiments (recommended)

Before running predictions, verify that all datasets are present and readable:

```bash
python -m funmirbench.cli.validate_experiments
```

This checks that:
- all `data_path` entries exist locally
- DE tables are readable
- gene identifiers can be detected robustly

---

### 4. Generate prediction scores

Run a prediction tool (currently a deterministic mock predictor) to produce
canonical miRNA–gene scores:

```bash
python -m funmirbench.cli.build_predictions --tool mock
```

This writes a canonical TSV under:

```
data/predictions/mock/
```

---

### 5. Register prediction tools

Index the prediction tool metadata:

```bash
python -m funmirbench.cli.build_predictions_index
```

This produces:

- `metadata/predictions.json`

---

### 6. Join experiments with predictions

The low-level join CLI can combine one dataset with one or more tools:

```bash
python -m funmirbench.cli.join_experiment_predictions \
  --dataset-id 001 \
  --tool mock \
  --combined \
  --out data/joined/001.tsv
```

This produces a joined TSV containing:
- one row per GT gene
- DE statistics such as `logFC`, `FDR`, and optional `PValue`
- one score column per selected tool as `score_<tool_id>`

---

### 7. Run the benchmark

The recommended downstream entrypoint is:

```bash
python -m funmirbench.cli.run_benchmark --config benchmark.example.yaml
```

This consumes:
- `metadata/datasets.json`
- experiment GT TSVs
- `metadata/predictions.json`
- canonical predictor TSVs

and writes a structured run directory under `results/`.

If the `experiments` section omits both `dataset_ids` and all filters, the runner selects all datasets.

For a minimal runnable example:

```bash
python -m funmirbench.cli.run_benchmark \
  --config examples/dummy_benchmark/benchmark.yaml
```

### 8. Plot and evaluate joined datasets

The plotting CLI is a lower-level evaluation step, not just a single scatter plot:

```bash
python -m funmirbench.cli.plot_correlation \
  --joined-tsv results/run_example/joined/001.tsv \
  --out-dir data/plots
```

This produces:
- per-tool scatter / PR / ROC outputs
- dataset-level heatmaps
- evaluation reports
- metric summaries
