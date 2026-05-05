# GEO ingestion pipeline

This pipeline is the **first stage** of the FuNmiRBench experiment processing workflow.
It downloads raw FASTQ files from GEO/SRA (or locates local files) and automatically
generates YAML configuration files for the RNA-seq pipeline (`funmirbench-experiments`).

---

## Overview of the full pipeline

```
1. Fill in metadata/mirna_experiment_info.tsv
          ↓
2. Run geo_download.py   →   data/experiments/raw/{GSE}/{SRR}.fastq.gz
                         →   pipelines/experiments/configs/{dataset_id}.yaml  (auto-generated)
          ↓
3. Run funmirbench-experiments --config pipelines/experiments/configs/{dataset_id}.yaml
          ↓
4. DE table written to data/experiments/processed/{dataset_id}.tsv
```

---

## Step 1 — Fill in the metadata TSV

Edit `metadata/mirna_experiment_info.tsv`. The file already contains columns for miRNA
and study-level metadata. For each experiment you want to process, fill in the three
sample columns:

| Column | Description |
|---|---|
| `control_samples` | Comma-separated list of control sample identifiers |
| `condition_samples` | Comma-separated list of condition sample identifiers |
| `raw_data_dir` | *Optional.* Path to a local directory with pre-existing FASTQ files. Leave empty to download from GEO. |

### GEO mode (download from GEO/SRA)

Leave `raw_data_dir` empty and provide **GSM accession IDs** as sample identifiers:

```
control_samples:    GSM6437108,GSM6437109,GSM6437110
condition_samples:  GSM6437113,GSM6437114,GSM6437115
raw_data_dir:       (empty)
```

The script resolves each GSM to its SRR run(s) via the NCBI API, then downloads
FASTQ files. NCBI (`prefetch` + `fasterq-dump`) is tried first; ENA is used as fallback.

### Local mode (pre-existing files)

Set `raw_data_dir` to the directory containing your FASTQ files and provide
**sample base-names** (without extension) as identifiers:

```
control_samples:    ctrl_rep1,ctrl_rep2,ctrl_rep3
condition_samples:  treated_rep1,treated_rep2,treated_rep3
raw_data_dir:       /path/to/your/fastq/files
```

The script looks for:
- Single-end: `{raw_data_dir}/{name}.fastq.gz`
- Paired-end (auto-detected): `{raw_data_dir}/{name}_1.fastq.gz` + `{raw_data_dir}/{name}_2.fastq.gz`

Rows with empty `control_samples` or `condition_samples` are skipped automatically,
so you can fill in the TSV incrementally.

---

## Step 2 — Set up the environment

```bash
conda env create -f pipelines/geo/environment.yml
conda activate funmirbench-geo
```

---

## Step 3 — Run the pipeline

Run from the **repo root**:

```bash
python pipelines/geo/geo_download.py --tsv metadata/mirna_experiment_info.tsv
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--tsv` | *(required)* | Path to the metadata TSV |
| `--output` | `data/experiments/raw` | Where FASTQs are downloaded |
| `--config-output-dir` | `pipelines/experiments/configs` | Where YAML configs are written |
| `--threads` | `4` | Threads for `fasterq-dump` |

---

## Step 4 — Run the RNA-seq pipeline

For each experiment, a YAML config is auto-generated at:
```
pipelines/experiments/configs/{dataset_id}.yaml
```

Before running, download the shared genome reference (once per machine):
```bash
uv run funmirbench-experiments-download-examples
```

Then run the RNA-seq pipeline for each experiment:
```bash
conda activate funmirbench-experiments
uv run funmirbench-experiments --config pipelines/experiments/configs/{dataset_id}.yaml
```

The DE table is written to `data/experiments/processed/{dataset_id}.tsv`.

---

## Outputs

| Path | Description |
|---|---|
| `data/experiments/raw/{GSE}/{SRR}.fastq.gz` | Downloaded FASTQ files (GEO mode) |
| `data/experiments/raw/{GSE}/manifest.json` | Maps GSM → SRR → file paths and group (GEO mode) |
| `pipelines/experiments/configs/{dataset_id}.yaml` | Auto-generated RNA-seq pipeline config |
| `data/experiments/processed/{dataset_id}.tsv` | Final DE table (produced by RNA-seq pipeline) |

---

## Notes

- Downloaded FASTQ files and DE tables are **not tracked by git** (see `.gitignore`).
- The default genome reference is Ensembl v109 (GRCh38). To use a different reference,
  edit the generated YAML config before running the RNA-seq pipeline.
- Thread counts in the generated YAML can be adjusted to match your machine.
