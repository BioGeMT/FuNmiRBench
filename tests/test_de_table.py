"""Tests for funmirbench.utils.de_table."""

import pandas as pd
import pytest

from funmirbench.utils.de_table import (
    import_pandas_or_error,
    read_de_table,
    read_de_table_columns,
)


class TestImportPandasOrError:
    def test_returns_pandas(self):
        result = import_pandas_or_error()
        assert hasattr(result, "read_csv")

    def test_context_in_message(self):
        # Can't easily test the ImportError path without uninstalling pandas,
        # but we can verify the function works when pandas is present.
        result = import_pandas_or_error(context="test context")
        assert result is pd


class TestReadDeTable:
    def test_normal_tsv(self, tmp_tsv_factory):
        path = tmp_tsv_factory(
            "gene_id\tlogFC\tFDR\n"
            "ENSG00000000001\t1.5\t0.01\n"
            "ENSG00000000002\t-0.8\t0.05\n"
        )
        df = read_de_table(pd, path)
        assert list(df.columns) == ["gene_id", "logFC", "FDR"]
        assert len(df) == 2

    def test_unnamed_first_column_renamed(self, tmp_tsv_factory):
        # Simulate R-style output where first column has no name
        path = tmp_tsv_factory(
            "\tlogFC\tFDR\n"
            "ENSG00000000001\t1.5\t0.01\n"
        )
        df = read_de_table(pd, path)
        assert df.columns[0] == "gene_id"

    def test_whitespace_fallback(self, tmp_tsv_factory):
        # Tab is present in the header (passes _assert_tsv_header) but
        # the data uses whitespace alignment with multiple spaces.
        path = tmp_tsv_factory(
            "gene_name\tlogFC\tFDR\n"
            "ENSG00000000001  1.5  0.01\n"
        )
        # This should still be parseable by the tab parser since header is tab-separated
        df = read_de_table(pd, path)
        assert "gene_name" in df.columns

    def test_non_tsv_raises(self, tmp_tsv_factory):
        path = tmp_tsv_factory("this is not a tsv file at all")
        with pytest.raises(ValueError, match="tab-separated"):
            read_de_table(pd, path)

    def test_nonexistent_file_raises(self, tmp_path):
        path = tmp_path / "does_not_exist.tsv"
        with pytest.raises(FileNotFoundError):
            read_de_table(pd, path)


class TestReadDeTableColumns:
    def test_returns_column_names(self, tmp_tsv_factory):
        path = tmp_tsv_factory(
            "gene_id\tlogFC\tlogCPM\tF\tPValue\tFDR\n"
            "ENSG00000000001\t1.5\t3.2\t10.0\t0.001\t0.01\n"
        )
        cols = read_de_table_columns(pd, path)
        assert cols == ["gene_id", "logFC", "logCPM", "F", "PValue", "FDR"]
