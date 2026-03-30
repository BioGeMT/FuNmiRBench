"""Tests for funmirbench.de_table (reader + gene ID detection)."""

import pandas as pd

from funmirbench.de_table import extract_gene_ids, find_gene_id_column, read_de_table


class TestFindGeneIdColumn:
    def test_explicit_gene_id(self):
        df = pd.DataFrame({"gene_id": ["ENSG00000000001"], "logFC": [1.0]})
        assert find_gene_id_column(df) == "gene_id"

    def test_explicit_gene_name(self):
        df = pd.DataFrame({"gene_name": ["ENSG00000000001"], "logFC": [1.0]})
        assert find_gene_id_column(df) == "gene_name"

    def test_gene_id_preferred_over_gene_name(self):
        df = pd.DataFrame({
            "gene_id": ["ENSG00000000001"],
            "gene_name": ["TP53"],
            "logFC": [1.0],
        })
        assert find_gene_id_column(df) == "gene_id"

    def test_ensembl_heuristic(self):
        df = pd.DataFrame({
            "my_genes": ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"],
            "logFC": [1.0, -0.5, 0.3],
        })
        assert find_gene_id_column(df) == "my_genes"

    def test_ensembl_below_threshold_falls_back_to_first_col(self):
        df = pd.DataFrame({
            "mixed": ["ENSG00000000001", "not_a_gene", "also_not", "nope"],
            "logFC": [1.0, -0.5, 0.3, 0.1],
        })
        assert find_gene_id_column(df) == "mixed"

    def test_index_heuristic(self):
        df = pd.DataFrame(
            {"logFC": [1.0, -0.5]},
            index=["ENSG00000000001", "ENSG00000000002"],
        )
        assert find_gene_id_column(df) == "__index__"


class TestExtractGeneIds:
    def test_from_column(self):
        df = pd.DataFrame({
            "gene_id": ["ENSG00000000001", "ENSG00000000002"],
            "logFC": [1.0, -0.5],
        })
        assert extract_gene_ids(df) == ["ENSG00000000001", "ENSG00000000002"]

    def test_from_index(self):
        df = pd.DataFrame(
            {"logFC": [1.0, -0.5]},
            index=["ENSG00000000001", "ENSG00000000002"],
        )
        assert extract_gene_ids(df) == ["ENSG00000000001", "ENSG00000000002"]

    def test_drops_na(self):
        df = pd.DataFrame({
            "gene_id": ["ENSG00000000001", None, "ENSG00000000003"],
            "logFC": [1.0, -0.5, 0.3],
        })
        assert len(extract_gene_ids(df)) == 2


class TestReadDeTable:
    def test_normal_tsv(self, tmp_tsv_factory):
        path = tmp_tsv_factory(
            "gene_id\tlogFC\tFDR\n"
            "ENSG00000000001\t1.5\t0.01\n"
            "ENSG00000000002\t-0.8\t0.05\n"
        )
        df = read_de_table(path)
        assert list(df.columns) == ["gene_id", "logFC", "FDR"]
        assert len(df) == 2

    def test_unnamed_first_column_renamed(self, tmp_tsv_factory):
        path = tmp_tsv_factory(
            "\tlogFC\tFDR\n"
            "ENSG00000000001\t1.5\t0.01\n"
        )
        df = read_de_table(path)
        assert df.columns[0] == "gene_id"

    def test_nonexistent_file_raises(self, tmp_path):
        path = tmp_path / "does_not_exist.tsv"
        with __import__("pytest").raises(FileNotFoundError):
            read_de_table(path)
