"""Tests for experiment validation helpers."""

from pathlib import Path

from funmirbench.validate_experiments import resolve_de_table_path


def test_resolve_de_table_path_prefers_repo_root_cwd(tmp_path, monkeypatch):
    repo_root = tmp_path
    experiments_tsv = repo_root / "metadata" / "mirna_experiment_info.tsv"
    experiments_tsv.parent.mkdir(parents=True)
    experiments_tsv.write_text("id\tde_table_path\n", encoding="utf-8")

    de_table = repo_root / "data" / "experiments" / "processed" / "demo.tsv"
    de_table.parent.mkdir(parents=True)
    de_table.write_text("gene_id\tlogFC\tFDR\n", encoding="utf-8")

    monkeypatch.chdir(repo_root)

    resolved = resolve_de_table_path(
        "data/experiments/processed/demo.tsv",
        experiments_tsv=experiments_tsv,
    )

    assert resolved == de_table.resolve()


def test_resolve_de_table_path_uses_explicit_root(tmp_path):
    experiments_tsv = tmp_path / "metadata" / "mirna_experiment_info.tsv"
    experiments_tsv.parent.mkdir(parents=True)
    experiments_tsv.write_text("id\tde_table_path\n", encoding="utf-8")

    root = tmp_path / "repo"
    de_table = root / "data" / "experiments" / "processed" / "demo.tsv"
    de_table.parent.mkdir(parents=True)
    de_table.write_text("gene_id\tlogFC\tFDR\n", encoding="utf-8")

    resolved = resolve_de_table_path(
        Path("data/experiments/processed/demo.tsv"),
        experiments_tsv=experiments_tsv,
        root=root,
    )

    assert resolved == de_table.resolve()
