# GEO ingestion pipeline

This pipeline is the **first stage** of the FuNmiRBench experiment processing workflow.
It downloads raw FASTQ files from GEO/SRA (or locates local FASTQ files) or verifies a
pre-existing count matrix and automatically generates YAML configuration files for
the RNA-seq pipeline (`funmirbench/experiments_pipeline.py`).

---

## Overview of the full pipeline

```
1. Fill in metadata/mirna_experiment_info.tsv
          ↓
2. Run geo_download.py   →   pipelines/experiments/configs/{dataset_id}.yaml  (auto-generated)
          ↓
3. Review and fine-tune the generated YAML config  ← important, see note below
          ↓
4. Run funmirbench/experiments_pipeline.py --config pipelines/experiments/configs/{dataset_id}.yaml
          ↓
5. DE table written to data/experiments/processed/{dataset_id}.tsv
```

---

## Step 1 — Fill in the metadata TSV

Edit `metadata/mirna_experiment_info.tsv`. For each experiment you want to process,
fill in the relevant columns depending on the mode:

| Column | Description |
|---|---|
| `control_samples` | Comma-separated list of control sample identifiers (GSM accessions, file base-names, or count matrix column names) |
| `condition_samples` | Comma-separated list of condition sample identifiers (same format as above) |
| `raw_data_dir` | *Reads mode only.* Path to a local directory with pre-existing FASTQ files. Leave empty to download from GEO. |
| `count_matrix_path` | *Count matrix mode only.* Path to the count matrix file. If set, count matrix mode is used. |
| `gene_id_column` | *Count matrix mode only.* Column name for gene IDs in the matrix (e.g. `ENSEMBL`). Required when `count_matrix_path` is set. |

Rows with empty `control_samples` or `condition_samples` are skipped automatically,
so you can fill in the TSV incrementally.

### GEO mode — download FASTQ files from GEO/SRA

Leave `raw_data_dir` and `count_matrix_path` empty, and provide **GSM accession IDs**
as sample identifiers:

```
control_samples:    GSM6437108,GSM6437109,GSM6437110
condition_samples:  GSM6437113,GSM6437114,GSM6437115
raw_data_dir:       (empty)
count_matrix_path:  (empty)
```

The script resolves each GSM to its SRR run(s) via the NCBI API, then downloads
FASTQ files. NCBI (`prefetch` + `fasterq-dump`) is tried first; ENA is used as fallback.

### Local reads mode — use pre-existing FASTQ files

Set `raw_data_dir` to the directory containing your FASTQ files and provide
**sample base-names** (without extension) as identifiers:

```
control_samples:    ctrl_rep1,ctrl_rep2,ctrl_rep3
condition_samples:  treated_rep1,treated_rep2,treated_rep3
raw_data_dir:       /path/to/your/fastq/files
count_matrix_path:  (empty)
```

The script looks for:
- Single-end: `{raw_data_dir}/{name}.fastq.gz`
- Paired-end (auto-detected): `{raw_data_dir}/{name}_1.fastq.gz` + `{raw_data_dir}/{name}_2.fastq.gz`

### Count matrix mode — use a pre-existing count matrix

Set `count_matrix_path` to the matrix file and provide the **column names** from
the matrix as sample identifiers:

```
control_samples:    HSVSMC_IP_miRCTRL_1,HSVSMC_IP_miRCTRL_2,HSVSMC_IP_miRCTRL_3
condition_samples:  HSVSMC_IP_MIR323_1,HSVSMC_IP_MIR323_2,HSVSMC_IP_MIR323_3
count_matrix_path:  data/experiments/raw/GSE253003/GSE253003_Count.csv.gz
gene_id_column:     ENSEMBL
raw_data_dir:       (empty)
```

The script only verifies that the matrix file exists and then generates the YAML config.
No downloading takes place.

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
python pipelines/geo/geo_download.py \
    --tsv metadata/mirna_experiment_info.tsv \
    --entrez-email your@email.com
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--tsv` | *(required)* | Path to the metadata TSV |
| `--entrez-email` | *(required)* | Email address for the NCBI Entrez API |
| `--output` | `data/experiments/raw` | Where FASTQs are downloaded (reads mode) |
| `--config-output-dir` | `pipelines/experiments/configs` | Where YAML configs are written |
| `--threads` | `4` | Threads for `fasterq-dump` |

---

## Step 4 — Review the generated YAML config

> **Before running the RNA-seq pipeline, always open and review the generated YAML.**

The YAML is auto-generated from the TSV and uses sensible defaults, but you should
verify and adjust the following before proceeding:

- **Genome reference paths** — the defaults point to Ensembl v109 (GRCh38). If your
  experiment uses a different genome or Ensembl version, update `genome_fasta_path`
  and `gtf_path` accordingly.
- **Thread counts** — `fastqc_threads`, `fastp_threads`, `star_threads`, and
  `featurecounts_threads` are set to fixed defaults. Adjust them to match the
  resources available on your machine.
- **Sample identifiers** — check that `sample_id` values and file paths look correct,
  especially for experiments with multiple SRR runs per GSM.
- **Metadata fields** — organism, cell line, tissue, and treatment are copied from
  the TSV; verify they are accurate before archiving results.

The generated YAML is at:
```
pipelines/experiments/configs/{dataset_id}.yaml
```

---

## Step 5 — Run the RNA-seq pipeline

Once the YAML config is reviewed and ready, refer to the [main project README](../../README.md)
for instructions on running the RNA-seq pipeline (`funmirbench/experiments_pipeline.py`).

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
- `--entrez-email` is required for GEO mode (NCBI policy); it is not used in local or count matrix mode but is still a required argument.
- If any SRR download fails, the entire experiment is skipped — no YAML config is written for it to avoid referencing missing files.
