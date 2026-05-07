"""Tests for the config-driven experiment ingestion workflow."""

import gzip
import json
import pathlib
import re
import subprocess

import pandas as pd
import pytest
import yaml

from funmirbench import experiments_pipeline


def write_gzip_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_run_count_matrix_mode_writes_benchmark_ready_table(tmp_path):
    counts = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "ctrl_a": [10, 100, 5],
            "ctrl_b": [10, 90, 5],
            "trt_a": [80, 100, 5],
            "trt_b": [75, 95, 5],
        }
    )
    counts_path = tmp_path / "counts.tsv"
    counts.to_csv(counts_path, sep="\t", index=False)

    config = {
        "gse": "GSE999999",
        "dataset_id": "manual_counts_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "count_matrix",
            "count_matrix_path": str(counts_path),
            "gene_id_column": "gene_id",
        },
        "comparison": {
            "control_columns": ["ctrl_a", "ctrl_b"],
            "treated_columns": ["trt_a", "trt_b"],
        },
        "metadata": {
            "organism": "Homo sapiens",
            "tested_cell_line": "HEK293T",
            "tissue": "Kidney",
            "method": "RNA-seq",
            "treatment": "miR-323a-3p",
            "article_pubmed_id": "https://pubmed.ncbi.nlm.nih.gov/12345678",
        },
    }
    config_path = tmp_path / "count_matrix.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def fake_subprocess_run(command, cwd, capture_output, text, check, env=None):
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is not None
        assert "Rscript" in command
        assert any(str(value).endswith("run_deseq2_counts.R") for value in command)

        output_path = pathlib.Path(command[command.index("--output") + 1])
        pd.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                "logFC": [2.5, 0.0, -0.1],
                "PValue": [0.001, 0.1, 0.6],
                "FDR": [0.003, 0.2, 0.8],
            }
        ).to_csv(output_path, sep="\t", index=False)
        return subprocess.CompletedProcess(command, 0, stdout="deseq2 ok\n", stderr="")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(experiments_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    de_table_path = pathlib.Path(result["de_table_path"])
    assert de_table_path.is_file()
    de_df = pd.read_csv(de_table_path, sep="\t")
    assert {"gene_id", "logFC", "PValue", "FDR"}.issubset(de_df.columns)
    assert de_df.set_index("gene_id").loc["ENSG1", "logFC"] > 0

    run_dir = pathlib.Path(result["run_dir"])
    assert (run_dir / "candidate_metadata.tsv").is_file()
    assert (run_dir / "sample_column_mapping.tsv").is_file()

    mapping_df = pd.read_csv(run_dir / "sample_column_mapping.tsv", sep="\t")
    assert set(mapping_df["resolved_count_column"]) == {"ctrl_a", "ctrl_b", "trt_a", "trt_b"}

    metadata_df = pd.read_csv(run_dir / "candidate_metadata.tsv", sep="\t")
    assert metadata_df.loc[0, "id"] == "manual_counts_demo"
    assert metadata_df.loc[0, "gse_url"] == "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE999999"

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "count_matrix"
    assert manifest["runtime"] == "local"
    assert manifest["gene_id_column"] == "gene_id"
    assert manifest["control_columns"] == ["ctrl_a", "ctrl_b"]
    assert manifest["treated_columns"] == ["trt_a", "trt_b"]
    assert any(str(value).endswith("run_deseq2_counts.R") for value in manifest["deseq2_command"])


def test_run_count_matrix_mode_requires_gene_id_column(tmp_path):
    counts = pd.DataFrame({"gene_id": ["ENSG1"], "ctrl": [10], "trt": [20]})
    counts_path = tmp_path / "counts.tsv"
    counts.to_csv(counts_path, sep="\t", index=False)

    config = {
        "dataset_id": "missing_gene_id_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "count_matrix",
            "count_matrix_path": str(counts_path),
            "gene_id_column": "",
        },
        "comparison": {
            "control_columns": ["ctrl"],
            "treated_columns": ["trt"],
        },
    }
    config_path = tmp_path / "missing_gene_id.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source.gene_id_column"):
        experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)


def test_run_count_matrix_mode_accepts_repo_root_relative_paths(tmp_path):
    counts_dir = tmp_path / "data" / "experiments" / "raw" / "GSE253003"
    counts_dir.mkdir(parents=True, exist_ok=True)
    counts_path = counts_dir / "counts.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "ctrl_1": [10, 20],
            "trt_1": [30, 10],
        }
    ).to_csv(counts_path, sep="\t", index=False)

    config_dir = tmp_path / "pipelines" / "experiments" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_id": "repo_relative_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "count_matrix",
            "count_matrix_path": "data/experiments/raw/GSE253003/counts.tsv",
            "gene_id_column": "gene_id",
        },
        "comparison": {
            "control_columns": ["ctrl_1"],
            "treated_columns": ["trt_1"],
        },
    }
    config_path = config_dir / "repo_relative.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def fake_subprocess_run(command, cwd, capture_output, text, check, env=None):
        del cwd
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is not None
        output_path = pathlib.Path(command[command.index("--output") + 1])
        pd.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2"],
                "logFC": [1.0, -1.0],
                "PValue": [0.01, 0.02],
                "FDR": [0.02, 0.03],
            }
        ).to_csv(output_path, sep="\t", index=False)
        return subprocess.CompletedProcess(command, 0, stdout="deseq2 ok\n", stderr="")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(experiments_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    assert pathlib.Path(result["de_table_path"]).is_file()


def test_run_reads_mode_runs_full_pipeline_and_writes_de_table(tmp_path):
    reads_files = {}
    for name in ["ctrl_a", "ctrl_b", "trt_a", "trt_b"]:
        path = tmp_path / f"{name}.fastq.gz"
        path.write_text("dummy\n", encoding="utf-8")
        reads_files[name] = path

    genome_fasta = tmp_path / "genome.fa.gz"
    write_gzip_text(genome_fasta, ">chr1\nAAAA\n")
    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "\n".join(
            [
                'chr1\tsrc\texon\t1\t4\t.\t+\t.\tgene_id "ENSG1"; transcript_id "tx1";',
                'chr1\tsrc\texon\t5\t8\t.\t+\t.\tgene_id "ENSG2"; transcript_id "tx2";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_subprocess_run(command, cwd, capture_output, text, check, env=None):
        del cwd
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is not None

        if command[0] == "fastqc":
            out_dir = pathlib.Path(command[command.index("--outdir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "report_fastqc.html").write_text("ok\n", encoding="utf-8")
            (out_dir / "report_fastqc.zip").write_text("ok\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="fastqc ok\n", stderr="")

        if command[0] == "fastp":
            reads_1_out = pathlib.Path(command[command.index("-o") + 1])
            reads_1_out.parent.mkdir(parents=True, exist_ok=True)
            reads_1_out.write_text("trimmed\n", encoding="utf-8")
            html_path = pathlib.Path(command[command.index("--html") + 1])
            json_path = pathlib.Path(command[command.index("--json") + 1])
            html_path.write_text("ok\n", encoding="utf-8")
            json_path.write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="fastp ok\n", stderr="")

        if command[0] == "STAR" and "--runMode" in command:
            out_dir = pathlib.Path(command[command.index("--genomeDir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "Genome").write_text("ok\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="index ok\n", stderr="")

        if command[0] == "STAR":
            out_prefix = pathlib.Path(command[command.index("--outFileNamePrefix") + 1])
            out_prefix.mkdir(parents=True, exist_ok=True)
            (out_prefix / "Aligned.sortedByCoord.out.bam").write_text("bam\n", encoding="utf-8")
            (out_prefix / "Log.final.out").write_text("star\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="star ok\n", stderr="")

        if command[0] == "featureCounts":
            output_path = pathlib.Path(command[command.index("-o") + 1])
            bam_paths = [value for value in command if str(value).endswith(".bam")]
            pd.DataFrame(
                {
                    "Geneid": ["ENSG1", "ENSG2"],
                    "Chr": ["chr1", "chr1"],
                    "Start": [1, 5],
                    "End": [4, 8],
                    "Strand": ["+", "+"],
                    "Length": [4, 4],
                    **{bam: [50, 10] for bam in bam_paths},
                }
            ).to_csv(output_path, sep="\t", index=False)
            return subprocess.CompletedProcess(command, 0, stdout="featurecounts ok\n", stderr="")

        if "Rscript" in command:
            sample_sheet = pathlib.Path(command[command.index("--counts") - 1]) if False else None
            del sample_sheet
            output_path = pathlib.Path(command[command.index("--output") + 1])
            pd.DataFrame(
                {
                    "gene_id": ["ENSG1", "ENSG2"],
                    "logFC": [2.5, -0.2],
                    "PValue": [0.001, 0.4],
                    "FDR": [0.003, 0.5],
                }
            ).to_csv(output_path, sep="\t", index=False)
            return subprocess.CompletedProcess(command, 0, stdout="deseq2 ok\n", stderr="")

        raise AssertionError(f"Unexpected command: {command}")

    config = {
        "dataset_id": "reads_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "reads",
            "genome_fasta_path": str(genome_fasta),
            "gtf_path": str(gtf_path),
            "fastqc_threads": 2,
            "fastp_threads": 2,
            "star_threads": 2,
            "featurecounts_threads": 2,
        },
        "comparison": {
            "control_samples": [
                {"sample_id": "ctrl_a", "reads_1": str(reads_files["ctrl_a"])},
                {"sample_id": "ctrl_b", "reads_1": str(reads_files["ctrl_b"])},
            ],
            "treated_samples": [
                {"sample_id": "trt_a", "reads_1": str(reads_files["trt_a"])},
                {"sample_id": "trt_b", "reads_1": str(reads_files["trt_b"])},
            ],
        },
        "metadata": {
            "organism": "Homo sapiens",
            "tested_cell_line": "HEK293T",
            "tissue": "Kidney",
            "method": "RNA-seq",
            "treatment": "miR-323a-3p",
        },
    }
    config_path = tmp_path / "reads.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(experiments_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    de_df = pd.read_csv(result["de_table_path"], sep="\t")
    assert {"gene_id", "logFC", "PValue", "FDR"}.issubset(de_df.columns)
    assert de_df.set_index("gene_id").loc["ENSG1", "logFC"] > 0

    run_dir = pathlib.Path(result["run_dir"])
    assert (run_dir / "reads_samples.tsv").is_file()
    assert (run_dir / "sample_column_mapping.tsv").is_file()
    assert (run_dir / "counts_matrix.tsv").is_file()
    assert (run_dir / "featurecounts_counts.tsv").is_file()
    assert (run_dir / "fastqc" / "raw" / "ctrl_a" / "report_fastqc.html").is_file()
    assert (run_dir / "fastqc" / "trimmed" / "ctrl_a" / "report_fastqc.html").is_file()
    assert (run_dir / "trimmed" / "ctrl_a" / "ctrl_a.fastq.gz").is_file()
    assert (run_dir / "star" / "ctrl_a" / "Aligned.sortedByCoord.out.bam").is_file()
    assert (run_dir / "reference" / "star_index" / "Genome").is_file()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "reads"
    assert manifest["runtime"] == "local"
    assert pathlib.Path(manifest["star_index"]) == run_dir / "reference" / "star_index"
    assert manifest["generated_star_index"] is True
    assert manifest["reused_star_index"] is False
    assert pathlib.Path(manifest["gtf_path"]) == gtf_path
    assert manifest["pipeline_stages"] == ["fastqc_raw", "fastp", "fastqc_trimmed", "star", "featurecounts", "deseq2"]
    assert set(manifest["sample_bams"]) == {"ctrl_a", "ctrl_b", "trt_a", "trt_b"}
    assert set(manifest["raw_fastqc_outputs"]) == {"ctrl_a", "ctrl_b", "trt_a", "trt_b"}
    assert set(manifest["trimmed_reads"]) == {"ctrl_a", "ctrl_b", "trt_a", "trt_b"}
    assert manifest["control_columns"] == ["ctrl_a", "ctrl_b"]
    assert manifest["treated_columns"] == ["trt_a", "trt_b"]


def test_run_reads_mode_reuses_existing_star_index_for_same_dataset(tmp_path):
    genome_fasta = tmp_path / "genome.fa.gz"
    write_gzip_text(genome_fasta, ">chr1\nAAAA\n")

    gtf_path = tmp_path / "genes.gtf.gz"
    write_gzip_text(
        gtf_path,
        'chr1\tsrc\texon\t1\t4\t.\t+\t.\tgene_id "ENSG1"; transcript_id "tx1";\n',
    )

    def fake_subprocess_run(command, cwd, capture_output, text, check, env=None):
        del cwd
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is not None

        if command[0] == "fastqc":
            out_dir = pathlib.Path(command[command.index("--outdir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "report_fastqc.html").write_text("ok\n", encoding="utf-8")
            (out_dir / "report_fastqc.zip").write_text("ok\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="fastqc ok\n", stderr="")

        if command[0] == "fastp":
            reads_1_out = pathlib.Path(command[command.index("-o") + 1])
            reads_1_out.parent.mkdir(parents=True, exist_ok=True)
            reads_1_out.write_text("trimmed\n", encoding="utf-8")
            html_path = pathlib.Path(command[command.index("--html") + 1])
            json_path = pathlib.Path(command[command.index("--json") + 1])
            html_path.write_text("ok\n", encoding="utf-8")
            json_path.write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="fastp ok\n", stderr="")

        if command[0] == "STAR" and "--runMode" in command:
            out_dir = pathlib.Path(command[command.index("--genomeDir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "Genome").write_text("ok\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="index ok\n", stderr="")

        if command[0] == "STAR":
            out_prefix = pathlib.Path(command[command.index("--outFileNamePrefix") + 1])
            out_prefix.mkdir(parents=True, exist_ok=True)
            (out_prefix / "Aligned.sortedByCoord.out.bam").write_text("bam\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="star ok\n", stderr="")

        if command[0] == "featureCounts":
            output_path = pathlib.Path(command[command.index("-o") + 1])
            bam_paths = [value for value in command if str(value).endswith(".bam")]
            pd.DataFrame(
                {
                    "Geneid": ["ENSG1", "ENSG2"],
                    "Chr": ["chr1", "chr1"],
                    "Start": [1, 5],
                    "End": [4, 8],
                    "Strand": ["+", "+"],
                    "Length": [4, 4],
                    **{bam: [40, 5] for bam in bam_paths},
                }
            ).to_csv(output_path, sep="\t", index=False)
            return subprocess.CompletedProcess(command, 0, stdout="featurecounts ok\n", stderr="")

        if "Rscript" in command:
            output_path = pathlib.Path(command[command.index("--output") + 1])
            pd.DataFrame(
                {
                    "gene_id": ["ENSG1", "ENSG2"],
                    "logFC": [1.7, -0.4],
                    "PValue": [0.01, 0.4],
                    "FDR": [0.02, 0.5],
                }
            ).to_csv(output_path, sep="\t", index=False)
            return subprocess.CompletedProcess(command, 0, stdout="deseq2 ok\n", stderr="")

        raise AssertionError(f"Unexpected command: {command}")

    config = {
        "dataset_id": "reads_reuse_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "reads",
            "genome_fasta_path": str(genome_fasta),
            "gtf_path": str(gtf_path),
            "fastqc_threads": 2,
            "fastp_threads": 2,
            "star_threads": 2,
            "featurecounts_threads": 2,
        },
        "comparison": {
            "control_samples": [{"sample_id": "ctrl_a", "reads_1": str(tmp_path / "ctrl_a.fastq.gz")}],
            "treated_samples": [{"sample_id": "trt_a", "reads_1": str(tmp_path / "trt_a.fastq.gz")}],
        },
        "metadata": {
            "organism": "Homo sapiens",
            "tested_cell_line": "HEK293T",
            "tissue": "Kidney",
            "method": "RNA-seq",
            "treatment": "miR-323a-3p",
        },
    }
    (tmp_path / "ctrl_a.fastq.gz").write_text("ctrl\n", encoding="utf-8")
    (tmp_path / "trt_a.fastq.gz").write_text("trt\n", encoding="utf-8")
    config_path = tmp_path / "reads_reuse.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiments_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(experiments_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    stamps = iter(["20260416_000001", "20260416_000002"])
    monkeypatch.setattr(experiments_pipeline, "utc_now_stamp", lambda: next(stamps))
    try:
        first = experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)
        result = experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    first_run_dir = pathlib.Path(first["run_dir"])
    run_dir = pathlib.Path(result["run_dir"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"] == "local"
    assert manifest["generated_star_index"] is False
    assert manifest["reused_star_index"] is True
    assert pathlib.Path(manifest["featurecounts_output"]).is_file()
    assert pathlib.Path(manifest["count_matrix_source"]).is_file()
    assert (first_run_dir / "reference" / "star_index" / "Genome").is_file()
    assert pathlib.Path(manifest["star_index"]) == first_run_dir / "reference" / "star_index"


def test_run_reads_mode_rejects_mixed_layouts(tmp_path):
    single = tmp_path / "single.fastq.gz"
    single.write_text("single\n", encoding="utf-8")
    paired_1 = tmp_path / "paired_1.fastq.gz"
    paired_2 = tmp_path / "paired_2.fastq.gz"
    paired_1.write_text("paired1\n", encoding="utf-8")
    paired_2.write_text("paired2\n", encoding="utf-8")

    genome_fasta = tmp_path / "genome.fa.gz"
    write_gzip_text(genome_fasta, ">chr1\nAAAA\n")
    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text('chr1\tsrc\texon\t1\t4\t.\t+\t.\tgene_id "ENSG1"; transcript_id "tx1";\n', encoding="utf-8")

    config = {
        "dataset_id": "mixed_layout_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "reads",
            "genome_fasta_path": str(genome_fasta),
            "gtf_path": str(gtf_path),
        },
        "comparison": {
            "control_samples": [{"sample_id": "ctrl", "reads_1": str(single)}],
            "treated_samples": [{"sample_id": "trt", "reads_1": str(paired_1), "reads_2": str(paired_2)}],
        },
    }
    config_path = tmp_path / "mixed_layout.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Mixed single-end and paired-end"):
        experiments_pipeline.run_ingestion_config(config_path, repo=tmp_path)
