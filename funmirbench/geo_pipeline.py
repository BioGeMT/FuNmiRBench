"""Config-driven experiment ingestion pipeline for benchmark-ready DE tables."""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
import yaml

GSE_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def gse_url(gse: str) -> str:
    return GSE_URL_TEMPLATE.format(gse=gse)


def ensure_clean_dir(path: pathlib.Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; rerun with --force to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: pathlib.Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(message, flush=True)


def load_yaml(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str, *, root: pathlib.Path, repo: pathlib.Path | None = None) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    relative_to_config = (root / path).resolve()
    if repo is None:
        return relative_to_config
    relative_to_repo = (repo / path).resolve()
    if relative_to_config.exists() or not relative_to_repo.exists():
        return relative_to_config
    return relative_to_repo


def download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def cached_download(url: str, dest: pathlib.Path, *, force: bool) -> pathlib.Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(download_bytes(url))
    return dest


def read_table_auto(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def output_de_table_rel_path(dataset_id: str) -> pathlib.Path:
    return pathlib.Path("data") / "experiments" / "processed" / f"{dataset_id}.tsv"


def require_fields(config: dict, fields: list[str]) -> None:
    missing = [field for field in fields if not normalize_space(config.get(field, ""))]
    if missing:
        raise ValueError(f"Config is missing required top-level fields: {missing}")


def open_text_auto(path: pathlib.Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def resolve_existing_source_path(
    source_cfg: dict, key: str, *, config_path: pathlib.Path, repo: pathlib.Path
) -> pathlib.Path | None:
    value = normalize_space(source_cfg.get(key, ""))
    if not value:
        return None
    path = resolve_path(value, root=config_path.parent, repo=repo)
    if not path.exists():
        raise ValueError(f"Config source.{key} does not exist: {path}")
    return path


def resolve_or_download_source_artifact(
    source_cfg: dict,
    *,
    path_key: str,
    url_key: str,
    config_path: pathlib.Path,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    force: bool,
    required: bool,
) -> tuple[pathlib.Path | None, str]:
    path_value = normalize_space(source_cfg.get(path_key, ""))
    url_value = normalize_space(source_cfg.get(url_key, ""))
    if path_value:
        path = resolve_path(path_value, root=config_path.parent, repo=repo)
        if not path.exists():
            raise ValueError(f"Config source.{path_key} does not exist: {path}")
        return path, ""
    if url_value:
        downloads_dir = run_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        dest = downloads_dir / pathlib.Path(url_value.split("?")[0]).name
        return cached_download(url_value, dest, force=force), url_value
    if required:
        raise ValueError(f"Config must include source.{path_key} or source.{url_key}.")
    return None, ""


def load_source_table(
    source_cfg: dict,
    *,
    path_key: str,
    url_key: str,
    run_dir: pathlib.Path,
    config_path: pathlib.Path,
    repo: pathlib.Path,
    force: bool,
) -> tuple[pd.DataFrame, pathlib.Path, str]:
    source_path, source_url = resolve_or_download_source_artifact(
        source_cfg,
        path_key=path_key,
        url_key=url_key,
        run_dir=run_dir,
        config_path=config_path,
        repo=repo,
        force=force,
        required=True,
    )
    assert source_path is not None
    return read_table_auto(source_path), source_path, source_url


def candidate_metadata_row(config: dict, *, de_table_rel_path: str) -> dict:
    gse = normalize_space(config.get("gse", ""))
    metadata_cfg = config.get("metadata", {})
    return {
        "id": config["dataset_id"],
        "mirna_name": config["mirna_name"],
        "mirna_sequence": metadata_cfg.get("mirna_sequence", ""),
        "article_pubmed_id": metadata_cfg.get("article_pubmed_id", ""),
        "organism": metadata_cfg.get("organism", ""),
        "tested_cell_line": metadata_cfg.get("tested_cell_line", ""),
        "treatment": metadata_cfg.get("treatment", ""),
        "tissue": metadata_cfg.get("tissue", ""),
        "method": metadata_cfg.get("method", "RNA-seq"),
        "experiment_type": config["experiment_type"],
        "gse_url": gse_url(gse) if gse else "",
        "de_table_path": de_table_rel_path,
    }


def write_candidate_metadata(config: dict, out_path: pathlib.Path, *, de_table_rel_path: str) -> None:
    pd.DataFrame([candidate_metadata_row(config, de_table_rel_path=de_table_rel_path)]).to_csv(
        out_path, sep="\t", index=False
    )


def validate_explicit_columns(df: pd.DataFrame, columns: list[str], *, key_name: str) -> list[str]:
    if not columns:
        raise ValueError(f"Config comparison.{key_name} is empty.")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Config comparison.{key_name} references missing columns: {missing}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"Config comparison.{key_name} contains duplicates: {columns}")
    return columns


def sample_entries_from_columns(columns: list[str], *, group_name: str) -> list[dict]:
    return [
        {
            "accession": column,
            "title": column,
            "group_label": group_name,
            "count_matrix_column": column,
        }
        for column in columns
    ]


def sample_column_mapping_rows(group_name: str, sample_entries: list[dict], columns: list[str]) -> list[dict]:
    rows = []
    for sample, column in zip(sample_entries, columns, strict=True):
        rows.append(
            {
                "group": group_name,
                "sample_accession": sample.get("accession", ""),
                "sample_title": sample.get("title", ""),
                "group_label": sample.get("group_label", ""),
                "resolved_count_column": column,
            }
        )
    return rows


def resolve_gene_id_column(source_cfg: dict, counts_df: pd.DataFrame) -> str:
    configured_gene_id = normalize_space(source_cfg.get("gene_id_column", ""))
    if not configured_gene_id:
        raise ValueError("Config source.gene_id_column is required for count-based modes.")
    if configured_gene_id not in counts_df.columns:
        raise ValueError(
            f"Configured source.gene_id_column {configured_gene_id!r} was not found in the count matrix."
        )
    return configured_gene_id


def require_local_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable {name!r} was not found on PATH. "
            "Install and activate the GEO environment from pipelines/geo/environment.yml."
        )


def run_logged_command(
    command: list[str],
    *,
    cwd: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    error_label: str,
) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{error_label} failed with exit code {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )


def run_deseq2_from_counts(
    *,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    counts_path: pathlib.Path,
    gene_id_column: str,
    control_cols: list[str],
    treated_cols: list[str],
    output_path: pathlib.Path,
) -> tuple[list[str], pathlib.Path, pathlib.Path]:
    require_local_binary("Rscript")
    log("Running DESeq2 from count matrix...")
    r_script = repo / "pipelines" / "geo" / "run_deseq2_counts.R"
    command = [
        "Rscript",
        str(r_script),
        "--counts",
        str(counts_path),
        "--gene-id-column",
        gene_id_column,
        "--control-columns",
        ",".join(control_cols),
        "--treated-columns",
        ",".join(treated_cols),
        "--output",
        str(output_path),
    ]
    stdout_path = run_dir / "deseq2.stdout.txt"
    stderr_path = run_dir / "deseq2.stderr.txt"
    run_logged_command(
        command,
        cwd=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        error_label="DESeq2 run",
    )
    if not output_path.exists():
        raise ValueError(f"DESeq2 completed but did not create the expected DE table: {output_path}")
    return command, stdout_path, stderr_path


def run_de_from_counts(
    *,
    config: dict,
    config_path: pathlib.Path,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    counts_df: pd.DataFrame,
    counts_source: pathlib.Path,
    count_matrix_url: str,
    source_mode: str,
    control_cols: list[str],
    treated_cols: list[str],
    control_samples: list[dict],
    treated_samples: list[dict],
    extra_manifest: dict | None = None,
) -> dict:
    overlap = sorted(set(control_cols) & set(treated_cols))
    if overlap:
        raise ValueError(f"Control and treated groups overlap: {overlap}")

    source_cfg = config.get("source", {})
    gene_id_col = resolve_gene_id_column(source_cfg, counts_df)
    mapping_rows = sample_column_mapping_rows("control", control_samples, control_cols) + sample_column_mapping_rows(
        "treated", treated_samples, treated_cols
    )
    pd.DataFrame(mapping_rows).to_csv(run_dir / "sample_column_mapping.tsv", sep="\t", index=False)

    de_results_path = run_dir / "de_results.tsv"
    deseq2_command, stdout_path, stderr_path = run_deseq2_from_counts(
        repo=repo,
        run_dir=run_dir,
        counts_path=counts_source,
        gene_id_column=gene_id_col,
        control_cols=control_cols,
        treated_cols=treated_cols,
        output_path=de_results_path,
    )
    de_df = read_table_auto(de_results_path)
    required = {"gene_id", "logFC", "FDR"}
    if not required.issubset(de_df.columns):
        raise ValueError(
            f"DESeq2 output is missing required columns. Expected at least {sorted(required)}, "
            f"found {sorted(de_df.columns)}"
        )

    out_rel = output_de_table_rel_path(config["dataset_id"])
    out_path = repo / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    de_df.to_csv(out_path, sep="\t", index=False)

    write_candidate_metadata(config, run_dir / "candidate_metadata.tsv", de_table_rel_path=str(out_rel))
    manifest = {
        "dataset_id": config["dataset_id"],
        "gse": normalize_space(config.get("gse", "")),
        "config_path": str(config_path),
        "source_mode": source_mode,
        "runtime": "local",
        "count_matrix_url": count_matrix_url,
        "count_matrix_source": str(counts_source),
        "count_matrix_rows": int(counts_df.shape[0]),
        "count_matrix_columns": int(counts_df.shape[1]),
        "gene_id_column": gene_id_col,
        "control_columns": control_cols,
        "treated_columns": treated_cols,
        "sample_column_mapping": mapping_rows,
        "deseq2_command": deseq2_command,
        "deseq2_stdout": str(stdout_path),
        "deseq2_stderr": str(stderr_path),
        "de_gene_count": int(de_df.shape[0]),
        "output_de_table": str(out_path),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    write_json(manifest, run_dir / "run_manifest.json")
    pd.DataFrame(control_samples + treated_samples).to_csv(run_dir / "samples.tsv", sep="\t", index=False)
    return {"run_dir": str(run_dir), "de_table_path": str(out_path)}


def normalize_sample_entry(sample: dict, *, group_name: str, root: pathlib.Path, repo: pathlib.Path) -> dict:
    sample_id = normalize_space(
        sample.get("sample_id") or sample.get("id") or sample.get("accession") or sample.get("sra_accession", "")
    )
    if not sample_id:
        raise ValueError(f"Each {group_name} sample must define sample_id or sra_accession.")

    reads_1 = ""
    reads_2 = ""
    reads_1_value = normalize_space(sample.get("reads_1", ""))
    reads_2_value = normalize_space(sample.get("reads_2", ""))
    sra_accession = normalize_space(sample.get("sra_accession", ""))

    if reads_1_value:
        resolved_reads_1 = resolve_path(reads_1_value, root=root, repo=repo)
        if not resolved_reads_1.exists():
            raise ValueError(f"reads_1 for sample {sample_id!r} does not exist: {resolved_reads_1}")
        reads_1 = str(resolved_reads_1)

        if reads_2_value:
            resolved_reads_2 = resolve_path(reads_2_value, root=root, repo=repo)
            if not resolved_reads_2.exists():
                raise ValueError(f"reads_2 for sample {sample_id!r} does not exist: {resolved_reads_2}")
            reads_2 = str(resolved_reads_2)
    elif not sra_accession:
        raise ValueError(f"Sample {sample_id!r} must provide reads_1 or sra_accession.")

    count_matrix_column = normalize_space(sample.get("count_matrix_column", "")) or sample_id
    return {
        "sample_id": sample_id,
        "group": group_name,
        "reads_1": reads_1,
        "reads_2": reads_2,
        "sra_accession": sra_accession,
        "count_matrix_column": count_matrix_column,
        "accession": sample_id,
        "title": sample_id,
        "group_label": group_name,
    }


def load_reads_samples(config: dict, *, config_path: pathlib.Path, repo: pathlib.Path) -> tuple[list[dict], list[dict]]:
    comparison_cfg = config.get("comparison", {})
    control_raw = comparison_cfg.get("control_samples", [])
    treated_raw = comparison_cfg.get("treated_samples", [])
    if not control_raw:
        raise ValueError("Config comparison.control_samples is empty.")
    if not treated_raw:
        raise ValueError("Config comparison.treated_samples is empty.")

    control_samples = [
        normalize_sample_entry(sample, group_name="control", root=config_path.parent, repo=repo)
        for sample in control_raw
    ]
    treated_samples = [
        normalize_sample_entry(sample, group_name="treated", root=config_path.parent, repo=repo)
        for sample in treated_raw
    ]
    sample_ids = [sample["sample_id"] for sample in control_samples + treated_samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Reads config contains duplicate sample_id values: {sample_ids}")
    return control_samples, treated_samples


def write_reads_sample_sheet(run_dir: pathlib.Path, control_samples: list[dict], treated_samples: list[dict]) -> pathlib.Path:
    sample_sheet_path = run_dir / "reads_samples.tsv"
    pd.DataFrame(control_samples + treated_samples).to_csv(sample_sheet_path, sep="\t", index=False)
    return sample_sheet_path


def salmon_quant_command(
    *,
    sample: dict,
    salmon_index: pathlib.Path,
    out_dir: pathlib.Path,
    library_type: str,
    extra_args: list[str],
    threads: int,
) -> list[str]:
    command = [
        "salmon",
        "quant",
        "-i",
        str(salmon_index),
        "-l",
        library_type,
        "-o",
        str(out_dir),
        "--threads",
        str(threads),
    ]
    reads_1 = sample["reads_1"]
    reads_2 = sample["reads_2"]
    if reads_2:
        command += ["-1", reads_1, "-2", reads_2]
    else:
        command += ["-r", reads_1]
    command.extend(extra_args)
    return command


def build_salmon_index(
    *,
    source_cfg: dict,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    transcript_fasta: pathlib.Path,
) -> tuple[pathlib.Path, list[str], pathlib.Path, pathlib.Path]:
    require_local_binary("salmon")
    log("Building Salmon index...")
    out_dir = run_dir / "reference" / "salmon_index"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    threads = int(source_cfg.get("salmon_threads", 1))
    extra_args = [str(arg) for arg in source_cfg.get("salmon_index_extra_args", [])]
    command = [
        "salmon",
        "index",
        "-t",
        str(transcript_fasta),
        "-i",
        str(out_dir),
        "--threads",
        str(threads),
        *extra_args,
    ]
    stdout_path = run_dir / "salmon_index.stdout.txt"
    stderr_path = run_dir / "salmon_index.stderr.txt"
    run_logged_command(
        command,
        cwd=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        error_label="Salmon index build",
    )
    if not out_dir.exists() or not any(out_dir.iterdir()):
        raise ValueError(f"Salmon index build completed but did not create files in {out_dir}")
    return out_dir, command, stdout_path, stderr_path


def parse_gtf_attributes(text: str) -> dict[str, str]:
    attributes = {}
    for part in [item.strip() for item in text.strip().strip(";").split(";") if item.strip()]:
        match = re.match(r"([A-Za-z0-9_.:-]+)\s+\"([^\"]+)\"", part)
        if match:
            attributes[match.group(1)] = match.group(2)
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            attributes[key.strip()] = value.strip().strip('"')
            continue
        bits = part.split(None, 1)
        if len(bits) == 2:
            attributes[bits[0]] = bits[1].strip().strip('"')
    return attributes


def generate_tx2gene_from_gtf(gtf_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    mapping = {}
    with open_text_auto(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = attrs.get("transcript_id") or attrs.get("transcriptId")
            gene_id = attrs.get("gene_id") or attrs.get("geneId")
            if transcript_id and gene_id:
                mapping[transcript_id] = gene_id
    if not mapping:
        raise ValueError(f"Could not derive transcript-to-gene mappings from {gtf_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"transcript_id": transcript_id, "gene_id": gene_id} for transcript_id, gene_id in sorted(mapping.items())]
    ).to_csv(out_path, sep="\t", index=False)
    return out_path


def prepare_reference_assets(
    *,
    source_cfg: dict,
    config_path: pathlib.Path,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    force: bool,
) -> dict:
    salmon_index = resolve_existing_source_path(source_cfg, "salmon_index", config_path=config_path, repo=repo)
    tx2gene_tsv = resolve_existing_source_path(source_cfg, "tx2gene_tsv", config_path=config_path, repo=repo)
    transcript_fasta_url = ""
    gtf_url = ""
    generated = {
        "generated_salmon_index": False,
        "generated_tx2gene_tsv": False,
    }
    command_paths = {}

    transcript_fasta = None
    if salmon_index is None:
        transcript_fasta, transcript_fasta_url = resolve_or_download_source_artifact(
            source_cfg,
            path_key="transcript_fasta_path",
            url_key="transcript_fasta_url",
            config_path=config_path,
            repo=repo,
            run_dir=run_dir,
            force=force,
            required=True,
        )
        assert transcript_fasta is not None
        salmon_index, index_command, index_stdout, index_stderr = build_salmon_index(
            source_cfg=source_cfg,
            repo=repo,
            run_dir=run_dir,
            transcript_fasta=transcript_fasta,
        )
        generated["generated_salmon_index"] = True
        command_paths.update(
            {
                "salmon_index_command": index_command,
                "salmon_index_stdout": str(index_stdout),
                "salmon_index_stderr": str(index_stderr),
            }
        )

    gtf_path = None
    if tx2gene_tsv is None:
        gtf_path, gtf_url = resolve_or_download_source_artifact(
            source_cfg,
            path_key="gtf_path",
            url_key="gtf_url",
            config_path=config_path,
            repo=repo,
            run_dir=run_dir,
            force=force,
            required=True,
        )
        assert gtf_path is not None
        tx2gene_tsv = generate_tx2gene_from_gtf(gtf_path, run_dir / "reference" / "tx2gene.tsv")
        generated["generated_tx2gene_tsv"] = True

    assert salmon_index is not None
    assert tx2gene_tsv is not None
    return {
        "salmon_index": salmon_index,
        "tx2gene_tsv": tx2gene_tsv,
        "transcript_fasta": str(transcript_fasta) if transcript_fasta else "",
        "transcript_fasta_url": transcript_fasta_url,
        "gtf_path": str(gtf_path) if gtf_path else "",
        "gtf_url": gtf_url,
        **generated,
        **command_paths,
    }


def run_salmon_quant(
    *,
    source_cfg: dict,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    sample: dict,
    salmon_index: pathlib.Path,
) -> pathlib.Path:
    require_local_binary("salmon")
    log(f"Running Salmon for sample {sample['sample_id']}...")
    out_dir = run_dir / "salmon" / sample["sample_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_args = [str(arg) for arg in source_cfg.get("salmon_extra_args", ["--validateMappings"])]
    library_type = normalize_space(source_cfg.get("library_type", "")) or "A"
    threads = int(source_cfg.get("salmon_threads", 1))
    command = salmon_quant_command(
        sample=sample,
        salmon_index=salmon_index,
        out_dir=out_dir,
        library_type=library_type,
        extra_args=extra_args,
        threads=threads,
    )
    run_logged_command(
        command,
        cwd=repo,
        stdout_path=run_dir / f"salmon_{sample['sample_id']}.stdout.txt",
        stderr_path=run_dir / f"salmon_{sample['sample_id']}.stderr.txt",
        error_label=f"Salmon quantification for sample {sample['sample_id']}",
    )
    quant_path = out_dir / "quant.sf"
    if not quant_path.exists():
        raise ValueError(f"Salmon completed but did not create {quant_path}")
    return quant_path


def detect_downloaded_fastqs(sample_reads_dir: pathlib.Path, accession: str) -> tuple[pathlib.Path, pathlib.Path | None]:
    paired_1 = sample_reads_dir / f"{accession}_1.fastq"
    paired_2 = sample_reads_dir / f"{accession}_2.fastq"
    single = sample_reads_dir / f"{accession}.fastq"
    if paired_1.exists() and paired_2.exists():
        return paired_1, paired_2
    if single.exists():
        return single, None
    fastqs = sorted(sample_reads_dir.glob("*.fastq"))
    if len(fastqs) == 1:
        return fastqs[0], None
    if len(fastqs) == 2:
        return fastqs[0], fastqs[1]
    raise ValueError(f"Could not detect FASTQ outputs for {accession} in {sample_reads_dir}")


def run_sra_download(
    *,
    source_cfg: dict,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    sample: dict,
) -> tuple[dict, list[str], pathlib.Path, pathlib.Path]:
    require_local_binary("prefetch")
    require_local_binary("fasterq-dump")
    accession = sample["sra_accession"]
    log(f"Downloading reads for sample {sample['sample_id']} from {accession}...")
    sample_reads_dir = run_dir / "reads" / sample["sample_id"]
    sample_reads_dir.mkdir(parents=True, exist_ok=True)
    sra_cache_dir = run_dir / "sra"
    sra_cache_dir.mkdir(parents=True, exist_ok=True)
    threads = int(source_cfg.get("sra_threads", 4))
    shell_command = (
        "set -euo pipefail; "
        f"prefetch --output-directory {shlex.quote(str(sra_cache_dir))} {shlex.quote(accession)}; "
        f"fasterq-dump --threads {threads} --split-files "
        f"--outdir {shlex.quote(str(sample_reads_dir))} {shlex.quote(accession)}"
    )
    command = ["bash", "-lc", shell_command]
    stdout_path = run_dir / f"sra_{sample['sample_id']}.stdout.txt"
    stderr_path = run_dir / f"sra_{sample['sample_id']}.stderr.txt"
    run_logged_command(
        command,
        cwd=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        error_label=f"SRA download for sample {sample['sample_id']}",
    )
    reads_1, reads_2 = detect_downloaded_fastqs(sample_reads_dir, accession)
    resolved = dict(sample)
    resolved["reads_1"] = str(reads_1)
    resolved["reads_2"] = str(reads_2) if reads_2 else ""
    return resolved, command, stdout_path, stderr_path


def materialize_reads_samples(
    *,
    source_cfg: dict,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    control_samples: list[dict],
    treated_samples: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    download_manifest = {
        "downloaded_samples": [],
        "sra_download_commands": {},
        "sra_download_stdout": {},
        "sra_download_stderr": {},
    }

    def resolve_sample(sample: dict) -> dict:
        if sample["reads_1"]:
            return sample
        resolved, command, stdout_path, stderr_path = run_sra_download(
            source_cfg=source_cfg,
            repo=repo,
            run_dir=run_dir,
            sample=sample,
        )
        download_manifest["downloaded_samples"].append(sample["sample_id"])
        download_manifest["sra_download_commands"][sample["sample_id"]] = command
        download_manifest["sra_download_stdout"][sample["sample_id"]] = str(stdout_path)
        download_manifest["sra_download_stderr"][sample["sample_id"]] = str(stderr_path)
        return resolved

    resolved_control = [resolve_sample(sample) for sample in control_samples]
    resolved_treated = [resolve_sample(sample) for sample in treated_samples]
    return resolved_control, resolved_treated, download_manifest


def write_salmon_manifest(
    run_dir: pathlib.Path,
    control_samples: list[dict],
    treated_samples: list[dict],
    quant_paths: dict[str, pathlib.Path],
) -> pathlib.Path:
    manifest_rows = []
    for sample in control_samples + treated_samples:
        manifest_rows.append(
            {
                "sample_id": sample["sample_id"],
                "condition": sample["group"],
                "quant_sf": str(quant_paths[sample["sample_id"]]),
            }
        )
    manifest_path = run_dir / "salmon_samples.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def run_deseq2(
    *,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    salmon_manifest: pathlib.Path,
    tx2gene_tsv: pathlib.Path,
    output_path: pathlib.Path,
) -> tuple[list[str], pathlib.Path, pathlib.Path]:
    require_local_binary("Rscript")
    log("Running tximport + DESeq2...")
    r_script = repo / "pipelines" / "geo" / "run_deseq2.R"
    command = [
        "Rscript",
        str(r_script),
        "--sample-sheet",
        str(salmon_manifest),
        "--tx2gene",
        str(tx2gene_tsv),
        "--output",
        str(output_path),
    ]
    stdout_path = run_dir / "deseq2.stdout.txt"
    stderr_path = run_dir / "deseq2.stderr.txt"
    run_logged_command(
        command,
        cwd=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        error_label="DESeq2 run",
    )
    if not output_path.exists():
        raise ValueError(f"DESeq2 completed but did not create the expected DE table: {output_path}")
    return command, stdout_path, stderr_path


def run_count_matrix_mode(
    config: dict,
    *,
    config_path: pathlib.Path,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    force: bool,
) -> dict:
    log(f"Mode: count_matrix ({config['dataset_id']})")
    source_cfg = config.get("source", {})
    log("Loading count matrix...")
    counts_df, counts_source, count_matrix_url = load_source_table(
        source_cfg,
        path_key="count_matrix_path",
        url_key="count_matrix_url",
        run_dir=run_dir,
        config_path=config_path,
        repo=repo,
        force=force,
    )

    comparison_cfg = config.get("comparison", {})
    control_cols = validate_explicit_columns(
        counts_df,
        [normalize_space(value) for value in comparison_cfg.get("control_columns", []) if str(value).strip()],
        key_name="control_columns",
    )
    treated_cols = validate_explicit_columns(
        counts_df,
        [normalize_space(value) for value in comparison_cfg.get("treated_columns", []) if str(value).strip()],
        key_name="treated_columns",
    )
    control_samples = sample_entries_from_columns(control_cols, group_name="control")
    treated_samples = sample_entries_from_columns(treated_cols, group_name="treated")
    return run_de_from_counts(
        config=config,
        config_path=config_path,
        repo=repo,
        run_dir=run_dir,
        counts_df=counts_df,
        counts_source=counts_source,
        count_matrix_url=count_matrix_url,
        source_mode="count_matrix",
        control_cols=control_cols,
        treated_cols=treated_cols,
        control_samples=control_samples,
        treated_samples=treated_samples,
    )


def run_reads_mode(
    config: dict,
    *,
    config_path: pathlib.Path,
    repo: pathlib.Path,
    run_dir: pathlib.Path,
    force: bool,
) -> dict:
    log(f"Mode: reads ({config['dataset_id']})")
    source_cfg = config.get("source", {})
    log("Loading reads config...")
    control_samples, treated_samples = load_reads_samples(config, config_path=config_path, repo=repo)
    control_samples, treated_samples, download_manifest = materialize_reads_samples(
        source_cfg=source_cfg,
        repo=repo,
        run_dir=run_dir,
        control_samples=control_samples,
        treated_samples=treated_samples,
    )
    sample_sheet = write_reads_sample_sheet(run_dir, control_samples, treated_samples)
    log("Preparing references...")
    reference_assets = prepare_reference_assets(
        source_cfg=source_cfg,
        config_path=config_path,
        repo=repo,
        run_dir=run_dir,
        force=force,
    )
    salmon_index = pathlib.Path(reference_assets["salmon_index"])
    tx2gene_tsv = pathlib.Path(reference_assets["tx2gene_tsv"])

    quant_paths = {}
    for sample in control_samples + treated_samples:
        quant_paths[sample["sample_id"]] = run_salmon_quant(
            source_cfg=source_cfg,
            repo=repo,
            run_dir=run_dir,
            sample=sample,
            salmon_index=salmon_index,
        )

    salmon_manifest = write_salmon_manifest(run_dir, control_samples, treated_samples, quant_paths)
    de_results_path = run_dir / "de_results.tsv"
    deseq2_command, stdout_path, stderr_path = run_deseq2(
        repo=repo,
        run_dir=run_dir,
        salmon_manifest=salmon_manifest,
        tx2gene_tsv=tx2gene_tsv,
        output_path=de_results_path,
    )
    de_df = read_table_auto(de_results_path)
    required = {"gene_id", "logFC", "FDR"}
    if not required.issubset(de_df.columns):
        raise ValueError(
            f"DESeq2 output is missing required columns. Expected at least {sorted(required)}, "
            f"found {sorted(de_df.columns)}"
        )

    out_rel = output_de_table_rel_path(config["dataset_id"])
    out_path = repo / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    de_df.to_csv(out_path, sep="\t", index=False)
    write_candidate_metadata(config, run_dir / "candidate_metadata.tsv", de_table_rel_path=str(out_rel))
    write_json(
        {
            "dataset_id": config["dataset_id"],
            "gse": normalize_space(config.get("gse", "")),
            "config_path": str(config_path),
            "source_mode": "reads",
            "runtime": "local",
            "salmon_index": str(salmon_index),
            "tx2gene_tsv": str(tx2gene_tsv),
            "reads_sample_sheet": str(sample_sheet),
            "salmon_manifest": str(salmon_manifest),
            "deseq2_command": deseq2_command,
            "deseq2_stdout": str(stdout_path),
            "deseq2_stderr": str(stderr_path),
            "sample_quant_dirs": {sample_id: str(path.parent) for sample_id, path in quant_paths.items()},
            "downloaded_samples": download_manifest["downloaded_samples"],
            "sra_download_commands": download_manifest["sra_download_commands"],
            "sra_download_stdout": download_manifest["sra_download_stdout"],
            "sra_download_stderr": download_manifest["sra_download_stderr"],
            "generated_salmon_index": reference_assets["generated_salmon_index"],
            "generated_tx2gene_tsv": reference_assets["generated_tx2gene_tsv"],
            "transcript_fasta": reference_assets["transcript_fasta"],
            "transcript_fasta_url": reference_assets["transcript_fasta_url"],
            "gtf_path": reference_assets["gtf_path"],
            "gtf_url": reference_assets["gtf_url"],
            "salmon_index_command": reference_assets.get("salmon_index_command", []),
            "salmon_index_stdout": reference_assets.get("salmon_index_stdout", ""),
            "salmon_index_stderr": reference_assets.get("salmon_index_stderr", ""),
            "output_de_table": str(out_path),
        },
        run_dir / "run_manifest.json",
    )
    pd.DataFrame(control_samples + treated_samples).to_csv(run_dir / "samples.tsv", sep="\t", index=False)
    return {"run_dir": str(run_dir), "de_table_path": str(out_path)}


def run_ingestion_config(config_path: pathlib.Path, repo: pathlib.Path | None = None, *, force: bool = False) -> dict:
    config_path = config_path.expanduser().resolve()
    repo = (repo or repo_root()).resolve()
    config = load_yaml(config_path)
    require_fields(config, ["dataset_id", "mirna_name", "experiment_type"])

    dataset_id = config["dataset_id"]
    run_dir = repo / "pipelines" / "geo" / "runs" / f"{utc_now_stamp()}_{dataset_id}"
    ensure_clean_dir(run_dir, force=force)
    shutil.copy2(config_path, run_dir / "config.yaml")
    log(f"Run dir: {run_dir}")

    source_cfg = config.get("source", {})
    mode = normalize_space(source_cfg.get("mode", ""))
    if not mode:
        raise ValueError("Config source.mode is required.")
    if mode == "count_matrix":
        return run_count_matrix_mode(config, config_path=config_path, repo=repo, run_dir=run_dir, force=force)
    if mode == "reads":
        return run_reads_mode(config, config_path=config_path, repo=repo, run_dir=run_dir, force=force)
    raise NotImplementedError(f"Unsupported source.mode {mode!r}.")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Run one experiment-ingestion config.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_ingestion_config(args.config, force=args.force)
    print(f"Run dir: {result['run_dir']}")
    print(f"DE table: {result['de_table_path']}")


if __name__ == "__main__":
    main()
