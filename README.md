# FuNmiRBench

Benchmarking framework for **functional miRNA target prediction**.

FuNmiRBench provides:

- Standardized metadata for >50 GEO functional miRNA perturbation datasets  
- A unified Python API to:
  - list and filter experiments by miRNA, cell line, perturbation type, tissue, GEO accession  
  - load differential expression tables (edgeR outputs) into pandas  
  - summarize the available datasets
- A foundation for baseline models and future dashboards (visualization & evaluation)
........
---

## 🔧 Installation (development setup)

Clone the repo:

```bash
git clone git@github.com:BioGeMT/FuNmiRBench.git
cd FuNmiRBench
```
Create and activate a conda env (example):
```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda create -n funmirbench python=3.12 pandas -y
conda activate funmirbench
```

Make the code discoverable:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```
📂 Project structure
```
FuNmiRBench/
├── data/
│   ├── processed_GEO/           # DE tables (not in git)
│   ├── raw_GEO/                 # raw / counts / etc (not in git)
│   └── predictions/             # precomputed tool scores (not in git)
│       └── tool_name.tsv        # e.g. targetscan.tsv
│
├── metadata/
│   ├── datasets_core.json       # curated DIANA/Zenodo datasets
│   ├── datasets_user.json       # user-registered datasets (optional)
│   ├── datasets.json            # combined view (core + user) - auto-built
│   ├── mirna_experiment_info.tsv
│   └── predictions.json         # metadata about prediction files
│
├── pipelines/
│   └── geo/
│       ├── README.md            # how the GEO -> DE pipeline works
│       ├── env.yml / env.R      # environment spec
│       ├── run_pipeline.sh      # entry point script
│
├── scripts/
│   ├── build_index.py           # builds metadata/datasets*.json
│   └── build_predictions_index.py  # builds metadata/predictions.json (future)
│
├── src/
│   └── funmirbench/
│       ├── __init__.py
│       ├── datasets.py          # everything about DE tables & metadata
│       ├── predictions.py       # everything about prediction tool scores
│       ├── models/              # baseline models (seed, alignment, etc.)
│       └── evaluation/          # correlation, PR curves, plots
│
└── tests/
    └── test_datasets.py, test_predictions.py, ...
```
📊 Using the dataset API

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
