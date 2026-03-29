"""Tests for the config-driven experiment ingestion workflow."""

import json
import pathlib
import re
import subprocess

import pandas as pd
import pytest
import yaml

from funmirbench import geo_pipeline


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
            "count_matrix_url": "",
            "gene_id_column": "gene_id",
            "min_total_count": 1,
            "prior_count": 1.0,
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
    monkeypatch.setattr(geo_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(geo_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = geo_pipeline.run_ingestion_config(config_path, repo=tmp_path)
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
        geo_pipeline.run_ingestion_config(config_path, repo=tmp_path)


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

    config_dir = tmp_path / "pipelines" / "geo" / "configs"
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
    monkeypatch.setattr(geo_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(geo_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = geo_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    assert pathlib.Path(result["de_table_path"]).is_file()


def test_run_reads_mode_builds_counts_and_writes_de_table(tmp_path):
    reads_files = {}
    for name in ["ctrl_a", "ctrl_b", "trt_a", "trt_b"]:
        path = tmp_path / f"{name}.fastq.gz"
        path.write_text("dummy\n", encoding="utf-8")
        reads_files[name] = path

    salmon_index = tmp_path / "salmon_index"
    salmon_index.mkdir()
    (salmon_index / "versionInfo.json").write_text("{}", encoding="utf-8")

    tx2gene_path = tmp_path / "tx2gene.tsv"
    pd.DataFrame(
        {
            "transcript_id": ["tx1", "tx2", "tx3"],
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
        }
    ).to_csv(tx2gene_path, sep="\t", index=False)

    def fake_subprocess_run(command, cwd, capture_output, text, check, env=None):
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is not None

        if "salmon" in command and "quant" in command:
            out_dir = pathlib.Path(command[command.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "Name": ["tx1", "tx2", "tx3"],
                    "Length": [1000, 1000, 1000],
                    "EffectiveLength": [900, 900, 900],
                    "TPM": [1.0, 2.0, 3.0],
                    "NumReads": [10, 20, 30],
                }
            ).to_csv(out_dir / "quant.sf", sep="\t", index=False)
            return subprocess.CompletedProcess(command, 0, stdout="salmon ok\n", stderr="")

        if "Rscript" in command:
            sample_sheet = pathlib.Path(command[command.index("--sample-sheet") + 1])
            output_path = pathlib.Path(command[command.index("--output") + 1])
            samples = pd.read_csv(sample_sheet, sep="\t")
            assert set(samples["condition"]) == {"control", "treated"}
            pd.DataFrame(
                {
                    "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                    "logFC": [2.5, 0.1, -0.2],
                    "PValue": [0.001, 0.1, 0.5],
                    "FDR": [0.003, 0.2, 0.7],
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
            "salmon_index": str(salmon_index),
            "tx2gene_tsv": str(tx2gene_path),
            "library_type": "A",
            "salmon_threads": 2,
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
    monkeypatch.setattr(geo_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(geo_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = geo_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    de_df = pd.read_csv(result["de_table_path"], sep="\t")
    assert {"gene_id", "logFC", "PValue", "FDR"}.issubset(de_df.columns)
    assert de_df.set_index("gene_id").loc["ENSG1", "logFC"] > 0

    run_dir = pathlib.Path(result["run_dir"])
    assert (run_dir / "reads_samples.tsv").is_file()
    assert (run_dir / "salmon_ctrl_a.stdout.txt").is_file()
    assert (run_dir / "salmon_ctrl_a.stderr.txt").is_file()
    assert (run_dir / "salmon" / "ctrl_a" / "quant.sf").is_file()
    assert (run_dir / "deseq2.stdout.txt").is_file()
    assert (run_dir / "deseq2.stderr.txt").is_file()
    assert (run_dir / "salmon_samples.tsv").is_file()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "reads"
    assert manifest["runtime"] == "local"
    assert pathlib.Path(manifest["salmon_index"]) == salmon_index
    assert pathlib.Path(manifest["tx2gene_tsv"]) == tx2gene_path
    assert set(manifest["sample_quant_dirs"]) == {"ctrl_a", "ctrl_b", "trt_a", "trt_b"}


def test_run_reads_mode_can_download_sra_and_prepare_references(tmp_path):
    transcript_fasta = tmp_path / "transcripts.fa"
    transcript_fasta.write_text(">tx1\nAAAA\n>tx2\nCCCC\n", encoding="utf-8")

    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "\n".join(
            [
                'chr1\tsrc\ttranscript\t1\t4\t.\t+\t.\tgene_id "ENSG1"; transcript_id "tx1";',
                'chr1\tsrc\ttranscript\t5\t8\t.\t+\t.\tgene_id "ENSG2"; transcript_id "tx2";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_subprocess_run(command, cwd, capture_output, text, check, env=None):
        assert capture_output is True
        assert text is True
        assert check is False
        assert env is not None

        joined = " ".join(str(part) for part in command)
        if "salmon index" in joined:
            out_dir = pathlib.Path(command[command.index("-i") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "versionInfo.json").write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="index ok\n", stderr="")

        if "prefetch" in joined and "fasterq-dump" in joined:
            shell = command[-1]
            outdir = pathlib.Path(re.search(r"--outdir ([^ ]+)", shell).group(1))
            accession = shell.rsplit(" ", 1)[-1]
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{accession}.fastq").write_text("@r1\nAAAA\n+\n####\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="sra ok\n", stderr="")

        if "salmon" in command and "quant" in command:
            out_dir = pathlib.Path(command[command.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "Name": ["tx1", "tx2"],
                    "Length": [1000, 1000],
                    "EffectiveLength": [900, 900],
                    "TPM": [5.0, 3.0],
                    "NumReads": [50, 20],
                }
            ).to_csv(out_dir / "quant.sf", sep="\t", index=False)
            return subprocess.CompletedProcess(command, 0, stdout="salmon ok\n", stderr="")

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
        "dataset_id": "reads_sra_demo",
        "mirna_name": "hsa-miR-323a-3p",
        "experiment_type": "OE",
        "source": {
            "mode": "reads",
            "transcript_fasta_path": str(transcript_fasta),
            "gtf_path": str(gtf_path),
            "library_type": "A",
            "salmon_threads": 2,
            "sra_threads": 2,
        },
        "comparison": {
            "control_samples": [{"sample_id": "ctrl_a", "sra_accession": "SRR000001"}],
            "treated_samples": [{"sample_id": "trt_a", "sra_accession": "SRR000002"}],
        },
        "metadata": {
            "organism": "Homo sapiens",
            "tested_cell_line": "HEK293T",
            "tissue": "Kidney",
            "method": "RNA-seq",
            "treatment": "miR-323a-3p",
        },
    }
    config_path = tmp_path / "reads_sra.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(geo_pipeline.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(geo_pipeline.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        result = geo_pipeline.run_ingestion_config(config_path, repo=tmp_path)
    finally:
        monkeypatch.undo()

    run_dir = pathlib.Path(result["run_dir"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"] == "local"
    assert manifest["generated_salmon_index"] is True
    assert manifest["generated_tx2gene_tsv"] is True
    assert manifest["downloaded_samples"] == ["ctrl_a", "trt_a"]
    assert (run_dir / "reference" / "tx2gene.tsv").is_file()
    assert (run_dir / "reference" / "salmon_index" / "versionInfo.json").is_file()
    assert (run_dir / "reads" / "ctrl_a" / "SRR000001.fastq").is_file()
    assert (run_dir / "reads" / "trt_a" / "SRR000002.fastq").is_file()
