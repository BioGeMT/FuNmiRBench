"""Consolidated DE-table reading and validation."""

from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)


def import_pandas_or_error(context: str = ""):
    """Lazy-import pandas, raising a clear error if not installed."""
    try:
        import pandas as pd
        return pd
    except ImportError as exc:
        msg = "pandas is required"
        if context:
            msg += f" for {context}"
        msg += "; install it with `pip install pandas`."
        raise ImportError(msg) from exc


def _assert_tsv_header(path: pathlib.Path) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if not first_line or "\t" not in first_line:
        raise ValueError(
            f"Expected a TSV (tab-delimited) DE table, but file does not "
            f"appear tab-separated: {path}"
        )


def _normalize_columns(columns, *, path: pathlib.Path) -> list[str]:
    out = [str(c).strip() for c in columns]
    if out and (out[0] == "" or out[0].startswith("Unnamed:")):
        logger.warning(
            "DE table %s has an empty/unnamed first column; "
            "using 'gene_id' as the column name.",
            path,
        )
        out[0] = "gene_id"
    return out


def read_de_table(pd, path: pathlib.Path):
    """
    Read a DE table as TSV.

    Normalizes column names (strips whitespace, renames unnamed first column
    to ``gene_id``).  Falls back to whitespace-delimited parsing when the
    tab-parse yields only a single column.
    """
    _assert_tsv_header(path)
    df = pd.read_csv(path, sep="\t")
    df.columns = _normalize_columns(df.columns, path=path)

    # Whitespace fallback: if tab-parse produced a single column and no
    # recognised gene column exists, re-parse with generic whitespace.
    if (
        len(df.columns) == 1
        and "gene_name" not in df.columns
        and "gene_id" not in df.columns
    ):
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        df.columns = _normalize_columns(df.columns, path=path)

    return df


def read_de_table_columns(pd, path: pathlib.Path) -> list[str]:
    """Read only the header row of a DE table and return column names."""
    _assert_tsv_header(path)
    df = pd.read_csv(path, sep="\t", nrows=0)
    columns = _normalize_columns(df.columns, path=path)
    if not columns:
        raise ValueError(
            "Could not detect a usable gene identifier column. Expected a TSV "
            "(tab-delimited) table with either a 'gene_id'/'gene_name' column "
            "or gene identifiers in the first column."
        )
    return columns
