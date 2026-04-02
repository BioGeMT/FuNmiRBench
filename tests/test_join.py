"""Tests for funmirbench.join."""

import pathlib
import textwrap

import pandas as pd
import pytest

from funmirbench import DatasetMeta
from funmirbench.join import build_joined, load_experiment_table


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


class TestLoadExperimentTable:
    def test_basic(self, tmp_path):
        de_path = tmp_path / "de.tsv"
        _write(de_path, (
            "gene_id\tlogFC\tFDR\tPValue\n"
            "ENSG001\t2.0\t0.001\t0.0001\n"
            "ENSG002\t-1.0\t0.05\t0.01\n"
        ))
        meta = DatasetMeta(
            id="T001", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE000", data_path="de.tsv", root=tmp_path,
        )
        out = load_experiment_table(meta)
        assert list(out.columns) == ["dataset_id", "mirna", "gene_id", "logFC", "FDR", "PValue"]
        assert len(out) == 2
        assert out["dataset_id"].iloc[0] == "T001"

    def test_index_genes(self, tmp_path):
        de_path = tmp_path / "de.tsv"
        _write(de_path, (
            "\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
        ))
        meta = DatasetMeta(
            id="T002", miRNA="hsa-miR-2", cell_line="A549",
            tissue="lung", perturbation="KD", organism="Homo sapiens",
            geo_accession="GSE001", data_path="de.tsv", root=tmp_path,
        )
        out = load_experiment_table(meta)
        assert "gene_id" in out.columns

    def test_missing_required_columns_raises(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\n"
            "ENSG001\t2.0\n"
        ))
        meta = DatasetMeta(
            id="T004", miRNA="hsa-miR-4", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE004", data_path="de.tsv", root=tmp_path,
        )
        with pytest.raises(ValueError, match="missing required columns"):
            load_experiment_table(meta)

    def test_duplicate_gene_ids_raise(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG001\t1.0\t0.010\n"
        ))
        meta = DatasetMeta(
            id="T005", miRNA="hsa-miR-5", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE005", data_path="de.tsv", root=tmp_path,
        )
        with pytest.raises(ValueError, match="Duplicate gene_id values"):
            load_experiment_table(meta)


class TestBuildJoined:
    def test_left_join(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
            "ENSG003\t0.5\t0.3\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "mirna\tgene_id\tscore\n"
            "hsa-miR-1\tENSG001\t0.9\n"
            "hsa-miR-1\tENSG002\t0.1\n"
        ))
        meta = DatasetMeta(
            id="T003", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE002", data_path="de.tsv", root=tmp_path,
        )
        predictions = {"mock": {"predictor_output_path": "scores.tsv"}}
        joined, paths = build_joined(meta, ["mock"], predictions, tmp_path)
        assert len(joined) == 3
        assert "score_mock" in joined.columns
        assert pd.isna(joined[joined["gene_id"] == "ENSG003"]["score_mock"].iloc[0])

    def test_left_join_targetscan_style_columns(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
            "ENSG003\t0.5\t0.3\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t0.9\n"
            "ENSG002\tGENE2\t\thsa-miR-1\t0.1\n"
        ))
        meta = DatasetMeta(
            id="T003B", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE002", data_path="de.tsv", root=tmp_path,
        )
        predictions = {"mock": {"predictor_output_path": "scores.tsv"}}
        joined, paths = build_joined(meta, ["mock"], predictions, tmp_path)
        assert len(joined) == 3
        assert "score_mock" in joined.columns
        assert pd.isna(joined[joined["gene_id"] == "ENSG003"]["score_mock"].iloc[0])

    def test_duplicate_tool_scores_raise(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "mirna\tgene_id\tscore\n"
            "hsa-miR-1\tENSG001\t0.9\n"
            "hsa-miR-1\tENSG001\t0.1\n"
        ))
        meta = DatasetMeta(
            id="T006", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE006", data_path="de.tsv", root=tmp_path,
        )
        predictions = {"mock": {"predictor_output_path": "scores.tsv"}}
        with pytest.raises(ValueError, match="Duplicate mirna\\+gene scores"):
            build_joined(meta, ["mock"], predictions, tmp_path)
