# FuNmiRBench

Benchmark functional miRNA target predictors against differential-expression tables.

## Install

Requirements:

- Python 3.10+
- `uv`
- `conda` for the GEO ingestion environment

```bash
python -m pip install uv
git clone git@github.com:BioGeMT/FuNmiRBench.git
cd FuNmiRBench
uv sync
```

For the GEO ingestion pipeline, create and activate the supported environment:

```bash
conda env create -f pipelines/geo/environment.yml
conda activate funmirbench-geo
```

## Repo Layout

Main directories:

- `data/experiments/processed/`: experiment DE tables used by the benchmark
- `data/experiments/raw/`: local raw GEO inputs such as count matrices and FASTQs
- `data/predictions/`: local generated predictor TSVs
- `metadata/mirna_experiment_info.tsv`: experiment registry
- `metadata/predictions_info.tsv`: predictor registry
- `pipelines/geo/`: experiment-ingestion backend files and example configs
- `pipelines/standardized_predictors/`: predictor pipelines
- `results/`: benchmark outputs

The benchmark reads file paths from the two metadata TSVs. `data/` holds the real files. `results/`
is only for benchmark output.

## Workflow

### 1. Generate Experiment Data

The experiment-ingestion pipeline creates the same DE tables that the benchmark consumes from
`data/experiments/processed/`.

Supported experiment inputs:

- count matrix:
  counts matrix + control columns + treated columns -> DESeq2
- reads:
  local FASTQs or SRA accessions + explicit sample groups -> `salmon + tximport + DESeq2`

This path expects the `funmirbench-geo` environment from `pipelines/geo/environment.yml` to be
active so `salmon`, `prefetch`, `fasterq-dump`, and `Rscript` are available on `PATH`.

Download the shipped real example inputs:

```bash
uv run funmirbench-geo-download-examples
```

Run the real count-matrix example:

```bash
uv run funmirbench-geo --config pipelines/geo/configs/gse253003.count_matrix.example.yaml
```

Tracked example configs:

- `pipelines/geo/configs/gse253003.count_matrix.example.yaml`
- `pipelines/geo/configs/gse93717.reads.example.yaml`

Reads configs can either:

- use local `reads_1` and `reads_2`
- use `sra_accession` and let the pipeline download reads

Reads configs can also either:

- use prebuilt `salmon_index` and `tx2gene_tsv`
- or build them from `transcript_fasta_path` and `gtf_path`

Each run writes:

- `data/experiments/processed/<dataset_id>.tsv`
- `pipelines/geo/runs/<timestamp>_<dataset_id>/candidate_metadata.tsv`
- `pipelines/geo/runs/<timestamp>_<dataset_id>/run_manifest.json`

### 2. Sync Experiment Metadata

The ingestion pipeline does not edit `metadata/mirna_experiment_info.tsv` by itself. It writes a
candidate row first. Then sync it into the registry with:

```bash
uv run funmirbench-sync-metadata --kind experiments
```

### 3. Generate Predictor Data

Predictor score files live under `data/predictions/` and are discovered through
`metadata/predictions_info.tsv`.

The repo ships two demo predictor pipelines:

```bash
uv run pipelines/standardized_predictors/mock/pipeline.py
uv run pipelines/standardized_predictors/cheating/pipeline.py
```

This creates:

- `data/predictions/mock/mock_canonical.tsv`
- `data/predictions/cheating/cheating_canonical.tsv`

The demo predictors already have registry rows in `metadata/predictions_info.tsv`.

### 4. Run The Benchmark

The default benchmark config is `benchmark.yaml`.

It already points at:

- `metadata/mirna_experiment_info.tsv`
- `metadata/predictions_info.tsv`

and ships with 3 real experiment datasets plus 2 demo predictors selected.

Run it with:

```bash
uv run funmirbench --config benchmark.yaml
```

All YAML paths are relative to the YAML file itself.

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

out_dir: results/
```

Filter behavior:

- different keys are combined with AND
- values inside one list are combined with OR

`benchmark.yaml` already includes other experiment and predictor columns as commented rows, so the
normal workflow is just to edit or uncomment filters.

## Outputs

After a benchmark run, `results/` contains:

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
uv run funmirbench-import-experiments --from-dir /path/to/tables
uv run funmirbench-validate-experiments --experiments-tsv metadata/mirna_experiment_info.tsv
uv run funmirbench-geo-download-examples
uv run funmirbench-geo --config config.yaml
uv run funmirbench-sync-metadata --kind experiments
uv run pytest
```
