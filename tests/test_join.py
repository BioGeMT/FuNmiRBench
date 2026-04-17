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
        assert list(out.columns) == ["dataset_id", "mirna", "perturbation", "gene_id", "logFC", "FDR", "PValue"]
        assert len(out) == 2
        assert out["dataset_id"].iloc[0] == "T001"
        assert out["perturbation"].iloc[0] == "OE"

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
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t0.9\n"
            "ENSG002\tGENE2\t\thsa-miR-1\t0.1\n"
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
        assert "global_rank_mock" in joined.columns
        assert pd.isna(joined[joined["gene_id"] == "ENSG003"]["score_mock"].iloc[0])

    def test_duplicate_tool_scores_keep_strongest(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t0.9\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t0.1\n"
        ))
        meta = DatasetMeta(
            id="T006", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE006", data_path="de.tsv", root=tmp_path,
        )
        predictions = {"mock": {"predictor_output_path": "scores.tsv"}}
        joined, _ = build_joined(meta, ["mock"], predictions, tmp_path)
        scored = joined.set_index("gene_id")["score_mock"].to_dict()
        assert scored["ENSG001"] == 0.9
        ranked = joined.set_index("gene_id")["global_rank_mock"].to_dict()
        assert ranked["ENSG001"] == 1.0

    def test_lower_is_stronger_scores_are_inverted(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t-0.9\n"
            "ENSG002\tGENE2\t\thsa-miR-1\t-0.1\n"
        ))
        meta = DatasetMeta(
            id="T007", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE007", data_path="de.tsv", root=tmp_path,
        )
        predictions = {
            "targetscan": {
                "predictor_output_path": "scores.tsv",
                "score_direction": "lower_is_stronger",
            }
        }
        joined, _ = build_joined(meta, ["targetscan"], predictions, tmp_path)
        scored = joined.set_index("gene_id")["score_targetscan"].to_dict()
        assert scored["ENSG001"] == 0.9
        assert scored["ENSG002"] == 0.1
        ranked = joined.set_index("gene_id")["global_rank_targetscan"].to_dict()
        assert ranked["ENSG001"] == 1.0
        assert ranked["ENSG002"] == 0.0

    def test_invalid_score_direction_raises(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t0.9\n"
        ))
        meta = DatasetMeta(
            id="T008", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE008", data_path="de.tsv", root=tmp_path,
        )
        predictions = {
            "bad_tool": {
                "predictor_output_path": "scores.tsv",
                "score_direction": "sideways_is_stronger",
            }
        }
        with pytest.raises(ValueError, match="Unsupported score_direction"):
            build_joined(meta, ["bad_tool"], predictions, tmp_path)

    def test_duplicate_lower_is_stronger_scores_keep_strongest(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t-0.2\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t-0.9\n"
        ))
        meta = DatasetMeta(
            id="T009", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE009", data_path="de.tsv", root=tmp_path,
        )
        predictions = {
            "targetscan": {
                "predictor_output_path": "scores.tsv",
                "score_direction": "lower_is_stronger",
            }
        }
        joined, _ = build_joined(meta, ["targetscan"], predictions, tmp_path)
        assert joined.loc[0, "score_targetscan"] == 0.9
        assert joined.loc[0, "global_rank_targetscan"] == 1.0

    def test_tied_scores_get_same_global_rank(self, tmp_path):
        _write(tmp_path / "de.tsv", (
            "gene_id\tlogFC\tFDR\n"
            "ENSG001\t2.0\t0.001\n"
            "ENSG002\t-1.0\t0.05\n"
            "ENSG003\t0.5\t0.20\n"
        ))
        _write(tmp_path / "scores.tsv", (
            "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
            "ENSG900\tGENEX\t\thsa-miR-other\t0.2\n"
            "ENSG001\tGENE1\t\thsa-miR-1\t0.8\n"
            "ENSG002\tGENE2\t\thsa-miR-1\t0.8\n"
            "ENSG003\tGENE3\t\thsa-miR-1\t0.4\n"
            "ENSG901\tGENEY\t\thsa-miR-other\t1.0\n"
        ))
        meta = DatasetMeta(
            id="T010", miRNA="hsa-miR-1", cell_line="HeLa",
            tissue="cervix", perturbation="OE", organism="Homo sapiens",
            geo_accession="GSE010", data_path="de.tsv", root=tmp_path,
        )
        predictions = {"mock": {"predictor_output_path": "scores.tsv"}}
        joined, _ = build_joined(meta, ["mock"], predictions, tmp_path)
        ranked = joined.set_index("gene_id")["global_rank_mock"].to_dict()
        assert ranked["ENSG001"] == ranked["ENSG002"]
        assert ranked["ENSG001"] == pytest.approx(2.0 / 3.0)
        assert ranked["ENSG003"] == pytest.approx(1.0 / 3.0)
