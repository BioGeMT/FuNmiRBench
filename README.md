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

Generate the demo predictor outputs:

```bash
uv run pipelines/standardized_predictors/random/pipeline.py
uv run pipelines/standardized_predictors/cheating/pipeline.py
uv run pipelines/standardized_predictors/perfect/pipeline.py
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

and selects 3 real experiment datasets plus 4 demo predictors and TargetScan.

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
`candidate_metadata.tsv` under `pipelines/experiments/runs/<timestamp>_<dataset_id>/` first.
Then sync it into the registry with:

```bash
uv run funmirbench-sync-metadata
```

That command auto-discovers all `candidate_metadata.tsv` files under `pipelines/experiments/runs/`
and upserts them into the registry. Re-running is safe — existing rows with matching `id` values are
replaced, not duplicated.

To sync a specific file instead:

```bash
uv run funmirbench-sync-metadata --input pipelines/experiments/runs/<run_dir>/candidate_metadata.tsv
```

### 3. Add Predictor Data

Predictor score files live under `data/predictions/` and are discovered through
`metadata/predictions_info.tsv`.

The repo ships three demo predictor pipelines:

```bash
uv run pipelines/standardized_predictors/random/pipeline.py
uv run pipelines/standardized_predictors/cheating/pipeline.py
uv run pipelines/standardized_predictors/perfect/pipeline.py
```

This creates:

- `data/predictions/random/random_standardized.tsv`
- `data/predictions/random/random_3000_standardized.tsv`
- `data/predictions/cheating/cheating_standardized.tsv`
- `data/predictions/perfect/perfect_standardized.tsv`

The built-in demo predictors are intentionally different:

- `random`: deterministic random baseline over the full available miRNA-gene pairs
- `random_3000`: deterministic random baseline capped at 3000 genes per dataset
- `cheating`: demo-only directional scores informed by the benchmark DE tables
- `perfect`: dataset-aware oracle scores that exactly separate benchmark positives from negatives

`cheating` and `perfect` are threshold-sensitive demo predictors. Their standardized outputs are
generated against a specific `(FDR threshold, effect threshold)` pair, and the benchmark now checks
that those build thresholds match the current `evaluation` config before running.

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

During evaluation, each predictor is scored only on miRNA-gene pairs that exist in that
predictor's standardized file. Missing pairs are not filled with zero for metrics. Each run writes
coverage information to `tables/coverage_per_experiment.tsv`, and the per-predictor Markdown/PDF
reports also record total rows, scored rows, missing rows, and coverage. For per-dataset
heatmaps and agreement plots, FuNmiRBench uses a dataset-local tie-aware rank over the scored
rows. For cross-dataset rank-distribution plots, it keeps a separate global tie-aware rank
derived from each predictor's full standardized file. Predictor-agreement top fractions use an
exact top-k selection per predictor with a deterministic tie-break instead of a quantile
threshold. Combined PR, ROC, and GSEA comparison plots are computed on the common set of genes
scored by all compared predictors.

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
  tool_id: [random, random_3000, cheating, perfect, targetscan]

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

After a benchmark run, `results/` contains one new run folder whose name summarizes the selected
datasets, miRNAs, predictors, perturbation mix, cell-line count, and evaluation thresholds. For example:

- `results/tag-demo__datasets-gse109725-oe-mir-204-5p-gse118315-ko-mir-124-3p-plus1__mirnas-hsa-mir-204-5p-hsa-mir-124-3p-plus1__tools-random-cheating__pert-ko-oe__cell3__fdr0p05-effect1-top10pct/`

Inside each run folder you get:

- `README.md`: human-readable run guide and quick-start map for the output folder
- `REPORT.pdf`: main run-level PDF report with explanations and selected combined plots
- `datasets/<dataset_id>/joined.tsv`: joined DE + predictor score table for that dataset
- `datasets/<dataset_id>/plots/predictors/<tool_id>/`: per-tool plots for that dataset
- `datasets/<dataset_id>/plots/comparisons/`: multi-predictor comparison plots for that dataset
- `datasets/<dataset_id>/plots/heatmaps/`: dataset-level heatmaps
- `datasets/<dataset_id>/reports/`: per-dataset Markdown/PDF reports and correlation TSVs
- `tables/per_experiment/`: per-experiment metric tables
- `tables/combined/`: cross-dataset predictor summary table
- `plots/combined/metrics/`, `plots/combined/coverage/`, `plots/combined/ranks/`: cross-dataset comparison plots grouped by theme
- `summary.json`: run summary

When 2 or more predictors are selected, each dataset gets:

- one score-vs-expected-effect scatter per predictor
- one combined PR curve on common scored pairs
- one combined ROC curve on common scored pairs
- one algorithms-vs-genes heatmap
- one predictor-correlation heatmap

## Commands

```bash
uv run funmirbench --config benchmark.yaml
uv run funmirbench-experiments-store
uv run funmirbench-validate-experiments --experiments-tsv metadata/mirna_experiment_info.tsv
uv run funmirbench-experiments-download-examples
uv run funmirbench-experiments --config config.yaml
uv run funmirbench-sync-metadata
uv run pytest
```
