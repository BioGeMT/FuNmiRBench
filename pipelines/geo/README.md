# GEO ingestion pipeline

This folder documents the **current GEO-based ingestion workflow** for FuNmiRBench experiments.

At the moment, GEO is the only supported source of functional miRNA perturbation experiments,
but the pipeline is designed so that **additional experiment sources can be added later**
without changing the core benchmarking logic.

---

## Purpose

The GEO pipeline is responsible for:

1. Downloading raw expression data from GEO
2. Running differential expression analysis (e.g. edgeR)
3. Producing **processed DE tables** (TSV)
4. Placing those tables under `data/experiments/processed/` so they can be indexed by FuNmiRBench

The resulting DE tables are **not tracked by git**.

---

## Relation to metadata

This pipeline does **not** modify metadata directly.

Instead:

- Experiment metadata is curated in `metadata/mirna_experiment_info.tsv`
- DE table filenames produced by this pipeline must match the `de_table_path` column
- The dataset index is generated separately via:

```bash
python -m funmirbench.cli.build_experiments_index
```

This separation keeps metadata curation and data processing decoupled.

---

## Output format (processed DE tables)

Each processed DE table:

- is a tab-separated `.tsv` file
- corresponds to exactly one experiment
- must contain at least the following columns (names must match; order does not matter):

```text
gene_name    logFC    logCPM    F    PValue    FDR
```

See `data/README.md` for a detailed column specification.

---

## Running the pipeline

The concrete steps depend on the GEO study and analysis choices.
Typical execution is orchestrated via:

- `run_pipeline.sh`
- the accompanying R / conda environments

This README intentionally avoids prescribing a single execution path;
its role is to document **interfaces and expectations**, not enforce implementation details.
