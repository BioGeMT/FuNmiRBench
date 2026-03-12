"""
Shared DE-table parsing and lightweight validation helpers.

These helpers are intentionally small and reusable by multiple CLIs
(`validate_experiments`, `import_experiments`) so DE-table validation rules
can evolve in one place.
"""

from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)


def import_pandas_or_error(*, context: str = "DE-table validation"):
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(f"{context} requires pandas.") from exc
    return pd


def _assert_tsv_header(path: pathlib.Path) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if not first_line or "\t" not in first_line:
        raise ValueError(
            f"Expected a TSV (tab-delimited) DE table, but file does not appear tab-separated: {path}"
        )


def _normalize_columns(columns, *, path: pathlib.Path) -> list[str]:
    out = [str(c).strip() for c in columns]
    if out and (out[0] == "" or out[0].startswith("Unnamed:")):
        logger.warning(
            "DE table %s has an empty/unnamed first column; using 'gene_id' as the column name.",
            path,
        )
        out[0] = "gene_id"
    return out


def read_de_table(pd, path: pathlib.Path):
    """
    Read a DE table as TSV (tab-delimited).
    """
    _assert_tsv_header(path)
    df = pd.read_csv(path, sep="\t")
    df.columns = _normalize_columns(df.columns, path=path)
    return df


def read_de_table_columns(pd, path: pathlib.Path) -> list[str]:
    """
    Read DE-table header columns from a TSV path.
    """
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
