# FuNmiRBench

Benchmark functional miRNA target predictors against differential-expression tables.

## Install

Requirements:

- Python 3.10+
- `uv`
- `conda` for the experiments pipeline environment

Install `uv` on your machine first:

```bash
python -m pip install uv
```

Then clone the repo and install the Python package environment:

```bash
git clone git@github.com:BioGeMT/FuNmiRBench.git
cd FuNmiRBench
uv sync
```

If you want to use the experiments pipeline, create and activate the extra local environment after entering the repo:

```bash
conda env create -f pipelines/experiments/environment.yml
conda activate funmirbench-experiments
```

That environment also includes `uv`, so `uv run ...` keeps working after activation.

## Repo Layout

Main directories:

- `data/experiments/processed/`: root directory for processed experiment DE tables
- `data/experiments/processed/18745741/`: local cache for curated benchmark DE tables from Zenodo record `18745741`; the repo currently ships the 3 default benchmark TSVs here
- `data/experiments/raw/`: local raw GEO inputs such as count matrices and FASTQs
- `data/predictions/`: local generated predictor TSVs
- `metadata/mirna_experiment_info.tsv`: experiment registry
- `metadata/predictions_info.tsv`: predictor registry
- `pipelines/experiments/`: experiment-ingestion backend files and example configs
- `pipelines/standardized_predictors/`: predictor pipelines
- `results/`: benchmark outputs

The benchmark reads file paths from the two metadata TSVs. `data/` holds the real files. `results/`
is only for benchmark output.

## Quick Start

If you just want to run the benchmark, you do not need the experiments pipeline first. The repo
already includes:

- experiment metadata in `metadata/mirna_experiment_info.tsv`
- predictor metadata in `metadata/predictions_info.tsv`

Generate the two demo predictor outputs:

```bash
uv run pipelines/standardized_predictors/mock/pipeline.py
uv run pipelines/standardized_predictors/cheating/pipeline.py
```

Then run the default benchmark:

```bash
uv run funmirbench --config benchmark.yaml
```

Before benchmarking, `funmirbench` syncs the selected curated experiment DE tables from Zenodo
into `data/experiments/processed/18745741/` as needed. The repo currently ships the 3 TSVs used by
the default benchmark config, while other curated benchmark DE tables are treated as fetched local cache.

The default config already points at:

- `metadata/mirna_experiment_info.tsv`
- `metadata/predictions_info.tsv`

and selects 3 real experiment datasets plus 2 demo predictors.

## Workflow

### 1. Add Experiment Data

The experiment-ingestion pipeline creates DE tables under `data/experiments/processed/` for local
workflow use.

For the curated benchmark datasets tracked in `metadata/mirna_experiment_info.tsv`, the expected
workflow is different: those metadata rows stay versioned in the repo, and the corresponding DE
tables live under the local `data/experiments/processed/18745741/` cache. The repo currently ships
the 3 default benchmark TSVs there, and other curated tables are fetched from Zenodo when needed.

Experiment config summary:

- top-level:
  `dataset_id`, `mirna_name`, `experiment_type`, optional `gse`
- `source`:
  `mode: count_matrix` or `mode: reads`
- `comparison`:
  control vs treated columns or explicit control vs treated samples
- `metadata`:
  fields that will later be synced into `metadata/mirna_experiment_info.tsv`

Supported inputs:

- count matrix: counts matrix + control columns + treated columns -> DESeq2
- reads: local FASTQs + local reference files + explicit sample groups -> `FastQC + fastp + STAR + featureCounts + DESeq2`

This path expects the `funmirbench-experiments` environment from `pipelines/experiments/environment.yml` to be
active so `fastqc`, `fastp`, `STAR`, `featureCounts`, and `Rscript` are available on `PATH`.

Download the shipped real example inputs:

```bash
uv run funmirbench-experiments-download-examples
```

That downloader fetches:

- the real `GSE253003` count matrix
- the real `GSE93717` FASTQ files
- the shared Homo sapiens Ensembl v109 genome FASTA and GTF used by the reads example

Run the real count-matrix example:

```bash
uv run funmirbench-experiments --config pipelines/experiments/configs/gse253003.count_matrix.example.yaml
```

Run the reads example the same way:

```bash
uv run funmirbench-experiments --config pipelines/experiments/configs/gse93717.reads.example.yaml
```

Tracked example configs:

- `pipelines/experiments/configs/gse253003.count_matrix.example.yaml`
- `pipelines/experiments/configs/gse93717.reads.example.yaml`

Reads configs can either:

- use local `reads_1` and optional `reads_2`
- use local `genome_fasta_path` and `gtf_path`

So the practical reads flow is:

1. activate `funmirbench-experiments`
2. run `uv run funmirbench-experiments-download-examples`
3. run `uv run funmirbench-experiments --config pipelines/experiments/configs/gse93717.reads.example.yaml`

The shipped reads example now points at the downloaded Ensembl v109 reference source files under
`data/experiments/raw/refs/ensembl_v109/`, so it builds the derived STAR index automatically.

Each run writes:

- `data/experiments/processed/<dataset_id>.tsv`
- `pipelines/experiments/runs/<timestamp>_<dataset_id>/candidate_metadata.tsv`
- `pipelines/experiments/runs/<timestamp>_<dataset_id>/run_manifest.json`

The reads example uses a reproduced dataset id, `GSE93717_OE_miR_941_deseq2`, so syncing it creates
a separate variant instead of overwriting the curated `GSE93717_OE_miR_941` registry row.

### 2. Sync Experiment Metadata

The ingestion pipeline does not edit `metadata/mirna_experiment_info.tsv` by itself. It writes a
candidate row first. Then sync it into the registry with:

```bash
uv run funmirbench-sync-metadata --kind experiments
```

### 3. Add Predictor Data

Predictor score files live under `data/predictions/` and are discovered through
`metadata/predictions_info.tsv`.

The repo ships two demo predictor pipelines:

```bash
uv run pipelines/standardized_predictors/mock/pipeline.py
uv run pipelines/standardized_predictors/cheating/pipeline.py
```

This creates:

- `data/predictions/mock/mock_standardized.tsv`
- `data/predictions/cheating/cheating_standardized.tsv`

The demo predictors already have registry rows in `metadata/predictions_info.tsv`.

### 4. Run The Benchmark

The default benchmark config is `benchmark.yaml`.

Benchmark config summary:

- `experiments_tsv`: experiment metadata table
- `predictions_tsv`: predictor metadata table
- `experiments`: which experiment rows to include
- `predictors`: which predictor rows to include
- `evaluation`: thresholds and ranking settings
- `tags`: optional labels included in the per-run output folder name
- `out_dir`: results root directory; each benchmark run creates its own subfolder under this root

Run it with:

```bash
uv run funmirbench --config benchmark.yaml
```

That command automatically syncs only the experiment DE tables selected by your benchmark config
from Zenodo into the local `data/experiments/processed/18745741/` cache before joining predictions.

If you want to prefetch the full curated experiment cache yourself, you can also run:

```bash
uv run funmirbench-experiments-store
```

YAML paths can be:

- absolute paths
- relative to the YAML file
- repo-root-relative paths such as `data/...` and `metadata/...`

The default config shape is:

```yaml
experiments_tsv: metadata/mirna_experiment_info.tsv
predictions_tsv: metadata/predictions_info.tsv

experiments:
  id: [GSE109725_OE_miR_204_5p, GSE118315_KO_miR_124_3p, GSE210778_OE_miR_375_3p]

predictors:
  tool_id: [predictor_1, predictor_2]

evaluation:
  fdr_threshold: 0.05
  abs_logfc_threshold: 1.0
  predictor_top_fraction: 0.10

tags: [demo]

out_dir: results/
```

Filter behavior:

- different keys are combined with AND
- values inside one list are combined with OR

`benchmark.yaml` already includes other experiment and predictor columns as commented rows, so the
normal workflow is just to edit or uncomment filters.

## Outputs

After a benchmark run, `results/` contains one new run folder, for example:

- `results/tag-demo__exp3__pred2__oe1__ko1__kd0__cell3/`

Inside each run folder you get:

- `joined/`: joined DE + predictor score tables
- `tables/`: APS, PR-AUC, AUROC, and Spearman summary tables
- `plots/<dataset_id>/`: per-dataset plots
- `reports/`: per-predictor text reports and predictor-correlation TSVs
- `summary.json`: run summary

When 2 or more predictors are selected, each dataset gets:

- one score-vs-logFC scatter per predictor
- one combined PR curve
- one combined ROC curve
- one algorithms-vs-genes heatmap
- one predictor-correlation heatmap

## Commands

```bash
uv run funmirbench --config benchmark.yaml
uv run funmirbench-experiments-store
uv run funmirbench-validate-experiments --experiments-tsv metadata/mirna_experiment_info.tsv
uv run funmirbench-experiments-download-examples
uv run funmirbench-experiments --config config.yaml
uv run funmirbench-sync-metadata --kind experiments
uv run pytest
```
