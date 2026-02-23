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


def pick_gene_col(df) -> str:
    for candidate in ("gene_id", "gene_name"):
        if candidate in df.columns:
            return candidate
    return str(df.columns[0])


def gene_ids_detectable(df) -> bool:
    try:
        gene_col = pick_gene_col(df)
    except Exception:
        return False
    return gene_col in df.columns
