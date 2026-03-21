"""Tests for funmirbench.utils.gene_id."""

import pandas as pd
import pytest

from funmirbench.utils.gene_id import extract_gene_ids, find_gene_id_column


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
            "my_genes": [
                "ENSG00000000001",
                "ENSG00000000002",
                "ENSG00000000003",
            ],
            "logFC": [1.0, -0.5, 0.3],
        })
        assert find_gene_id_column(df) == "my_genes"

    def test_ensembl_heuristic_below_threshold(self):
        df = pd.DataFrame({
            "mixed": ["ENSG00000000001", "not_a_gene", "also_not", "nope"],
            "logFC": [1.0, -0.5, 0.3, 0.1],
        })
        with pytest.raises(ValueError, match="Could not identify"):
            find_gene_id_column(df)

    def test_index_heuristic(self):
        df = pd.DataFrame(
            {"logFC": [1.0, -0.5]},
            index=["ENSG00000000001", "ENSG00000000002"],
        )
        assert find_gene_id_column(df) == "__index__"

    def test_no_gene_ids_raises(self):
        df = pd.DataFrame({"logFC": [1.0], "FDR": [0.05]})
        with pytest.raises(ValueError, match="Could not identify"):
            find_gene_id_column(df)


class TestExtractGeneIds:
    def test_from_column(self):
        df = pd.DataFrame({
            "gene_id": ["ENSG00000000001", "ENSG00000000002"],
            "logFC": [1.0, -0.5],
        })
        ids = extract_gene_ids(df)
        assert ids == ["ENSG00000000001", "ENSG00000000002"]

    def test_from_index(self):
        df = pd.DataFrame(
            {"logFC": [1.0, -0.5]},
            index=["ENSG00000000001", "ENSG00000000002"],
        )
        ids = extract_gene_ids(df)
        assert ids == ["ENSG00000000001", "ENSG00000000002"]

    def test_drops_na(self):
        df = pd.DataFrame({
            "gene_id": ["ENSG00000000001", None, "ENSG00000000003"],
            "logFC": [1.0, -0.5, 0.3],
        })
        ids = extract_gene_ids(df)
        assert len(ids) == 2
        assert None not in ids
