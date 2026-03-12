"""
Shared DE-table parsing and lightweight validation helpers.

These helpers are intentionally small and reusable by multiple CLIs
(`validate_experiments`, `import_experiments`) so DE-table validation rules
can evolve in one place.
"""

from __future__ import annotations

import pathlib


def import_pandas_or_error(*, context: str = "DE-table validation"):
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(f"{context} requires pandas.") from exc
    return pd


def read_de_table(pd, path: pathlib.Path):
    """
    Read a DE table with tab delimiter first, then whitespace fallback.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    if "gene_name" not in df.columns and "gene_id" not in df.columns:
        df2 = pd.read_csv(path, sep=r"\s+", engine="python")
        df2.columns = [str(c).strip() for c in df2.columns]
        df = df2
    return df


def read_de_table_columns(pd, path: pathlib.Path) -> list[str]:
    """
    Read DE-table header columns from path (tab delimiter first, then whitespace fallback).
    """
    df = pd.read_csv(path, sep="\t", nrows=0)
    columns = [str(c).strip() for c in df.columns]
    if "gene_name" not in columns and "gene_id" not in columns:
        df2 = pd.read_csv(path, sep=r"\s+", engine="python", nrows=0)
        columns = [str(c).strip() for c in df2.columns]

    if not columns:
        raise ValueError("No columns available to detect gene identifier")
    return columns
