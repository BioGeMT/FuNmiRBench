# FuNmiRBench

Benchmarking framework for **functional miRNA target prediction**.

FuNmiRBench provides:

- Standardized metadata for >50 GEO functional miRNA perturbation datasets  
- A unified Python API to:
  - list and filter experiments by miRNA, cell line, perturbation type, tissue, GEO accession  
  - load differential expression tables (edgeR outputs) into pandas  
  - summarize the available datasets
- A foundation for baseline models and future dashboards (visualization & evaluation)

---

## 🔧 Installation (development setup)

Clone the repo:

```bash
git clone git@github.com:BioGeMT/FuNmiRBench.git
cd FuNmiRBench

Create and activate a conda env (example):

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda create -n funmirbench python=3.12 pandas -y
conda activate funmirbench


Make the code discoverable:

export PYTHONPATH="$PWD/src:$PYTHONPATH"

📂 Project structure
FuNmiRBench/
├── data/
│   ├── processed_GEO/   # GEO DE tables (NOT stored in git; see data/README.md)
│   └── raw_GEO/         # optional raw inputs (also NOT stored in git)
│
├── metadata/
│   ├── datasets.json            # metadata for all experiments (used by the API)
│   ├── index.json               # optional index (may be deprecated later)
│   └── mirna_experiment_info.tsv  # curated table describing each experiment
│
├── scripts/
│   └── build_index.py           # script to generate metadata/datasets.json
│
├── src/
│   └── funmirbench/
│       ├── __init__.py
│       ├── datasets.py          # main dataset API
│       ├── models/              # placeholder for baseline models
│       └── evaluation/          # placeholder for metrics & plots
│
└── tests/                       # tests (planned)

📊 Using the dataset API

Basic loading:

from funmirbench import load_dataset, load_all_datasets

# Load a single dataset by ID (e.g. "001")
df = load_dataset("001")

# Load all datasets
df_all = load_all_datasets()

# Filter by perturbation
df_oe = load_all_datasets(perturbation="overexpression")
df_kd = load_all_datasets(perturbation="knockdown")


Metadata exploration:

from funmirbench import datasets

datasets.list_mirnas()
datasets.list_cell_lines()
datasets.list_perturbations()
datasets.summarize_cell_lines()
datasets.summarize_mirnas()

summaries = datasets.summarize_datasets()
