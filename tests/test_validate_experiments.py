"""Tests for experiment validation helpers."""

import subprocess
import sys

import pandas as pd

from funmirbench.validate_experiments import resolve_de_table_path, validate_experiments


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _registry_row(de_table_path="data/experiments/processed/18745741/demo.tsv", **overrides):
    row = {
        "id": "T001",
        "mirna_name": "hsa-miR-test",
        "organism": "Homo sapiens",
        "experiment_type": "OE",
        "de_table_path": de_table_path,
    }
    row.update(overrides)
    return row


def _write_registry(tmp_path, rows):
    experiments_tsv = tmp_path / "metadata" / "mirna_experiment_info.tsv"
    experiments_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(experiments_tsv, sep="\t", index=False)
    return experiments_tsv


def _write_de_table(tmp_path, content):
    de_table = tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    _write(de_table, content)
    return de_table


def _valid_de_table():
    return (
        "gene_id\tlogFC\tFDR\n"
        "ENSG00000000001\t-2.0\t0.01\n"
        "ENSG00000000002\t0.2\t0.50\n"
    )


def _issue_checks(summary):
    return {issue.check for issue in summary.issues}


def test_resolve_de_table_path_prefers_repo_root_cwd(tmp_path, monkeypatch):
    repo_root = tmp_path
    _write(repo_root / "metadata" / "mirna_experiment_info.tsv", "id\tde_table_path\n")

    de_table = repo_root / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    de_table.parent.mkdir(parents=True)
    de_table.write_text("gene_id\tlogFC\tFDR\n", encoding="utf-8")

    monkeypatch.chdir(repo_root)

    resolved = resolve_de_table_path("data/experiments/processed/18745741/demo.tsv")

    assert resolved == de_table.resolve()


def test_resolve_de_table_path_uses_explicit_root(tmp_path):
    root = tmp_path / "repo"
    _write(root / "metadata" / "mirna_experiment_info.tsv", "id\tde_table_path\n")

    de_table = root / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    de_table.parent.mkdir(parents=True)
    de_table.write_text("gene_id\tlogFC\tFDR\n", encoding="utf-8")

    resolved = resolve_de_table_path(
        "data/experiments/processed/18745741/demo.tsv",
        root=root,
    )

    assert resolved == de_table.resolve()


def test_validate_experiments_accepts_benchmark_ready_table(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])
    _write_de_table(tmp_path, _valid_de_table())
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert summary.ok
    assert summary.total == 1
    assert summary.files_present == 1
    assert summary.benchmark_ready == 1


def test_validate_experiments_requires_registry_columns(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(
        tmp_path,
        [{"id": "T001", "de_table_path": "data/experiments/processed/18745741/demo.tsv"}],
    )
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert _issue_checks(summary) == {"registry_columns"}


def test_validate_experiments_rejects_duplicate_registry_values(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(
        tmp_path,
        [
            _registry_row(id="T001"),
            _registry_row(id="T001"),
        ],
    )
    _write_de_table(tmp_path, _valid_de_table())
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert {"duplicate_dataset_ids", "duplicate_de_table_paths"} <= _issue_checks(summary)


def test_validate_experiments_rejects_missing_de_table_file(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert "de_table_file" in _issue_checks(summary)


def test_validate_experiments_rejects_missing_required_de_columns(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])
    _write_de_table(
        tmp_path,
        "gene_id\tlogFC\n"
        "ENSG00000000001\t-2.0\n",
    )
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert "required_de_columns" in _issue_checks(summary)


def test_validate_experiments_rejects_duplicate_blank_and_non_ensembl_gene_ids(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])
    _write_de_table(
        tmp_path,
        "gene_id\tlogFC\tFDR\n"
        "ENSG00000000001\t-2.0\t0.01\n"
        "ENSG00000000001\t0.2\t0.50\n"
        "\t0.3\t0.60\n"
        "TP53\t0.4\t0.70\n",
    )
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert {"gene_id_values", "duplicate_gene_ids", "ensembl_gene_ids"} <= _issue_checks(summary)


def test_validate_experiments_rejects_non_numeric_values(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])
    _write_de_table(
        tmp_path,
        "gene_id\tlogFC\tFDR\n"
        "ENSG00000000001\tbad\t0.01\n"
        "ENSG00000000002\t0.2\tbad\n",
    )
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert {"numeric_logfc", "numeric_fdr"} <= _issue_checks(summary)


def test_validate_experiments_rejects_fdr_outside_open_closed_unit_interval(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])
    _write_de_table(
        tmp_path,
        "gene_id\tlogFC\tFDR\n"
        "ENSG00000000001\t-2.0\t0\n"
        "ENSG00000000002\t0.2\t1.20\n",
    )
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert "fdr_range" in _issue_checks(summary)


def test_validate_experiments_rejects_invalid_perturbation(tmp_path, monkeypatch):
    experiments_tsv = _write_registry(tmp_path, [_registry_row(experiment_type="ACTIVATION")])
    _write_de_table(tmp_path, _valid_de_table())
    monkeypatch.chdir(tmp_path)

    summary = validate_experiments(experiments_tsv)

    assert not summary.ok
    assert "experiment_type" in _issue_checks(summary)


def test_validate_experiments_requires_positive_and_negative_gt_classes(tmp_path, monkeypatch):
    zero_positive_root = tmp_path / "zero_positive"
    zero_positive_tsv = _write_registry(zero_positive_root, [_registry_row()])
    _write_de_table(
        zero_positive_root,
        "gene_id\tlogFC\tFDR\n"
        "ENSG00000000001\t0.2\t0.50\n"
        "ENSG00000000002\t0.3\t0.60\n",
    )
    monkeypatch.chdir(zero_positive_root)

    zero_positive = validate_experiments(zero_positive_tsv)
    assert not zero_positive.ok
    assert "no positive genes" in zero_positive.issues[0].message

    zero_negative_root = tmp_path / "zero_negative"
    zero_negative_tsv = _write_registry(zero_negative_root, [_registry_row()])
    _write_de_table(
        zero_negative_root,
        "gene_id\tlogFC\tFDR\n"
        "ENSG00000000001\t-2.0\t0.01\n"
        "ENSG00000000002\t-3.0\t0.02\n",
    )
    monkeypatch.chdir(zero_negative_root)

    zero_negative = validate_experiments(zero_negative_tsv)
    assert not zero_negative.ok
    assert "no negative genes" in zero_negative.issues[0].message


def test_validate_experiments_cli_returns_nonzero_on_failure(tmp_path):
    experiments_tsv = _write_registry(tmp_path, [_registry_row()])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "funmirbench.validate_experiments",
            "--experiments-tsv",
            str(experiments_tsv),
        ],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 1
    assert "Validation issues" in result.stdout + result.stderr
