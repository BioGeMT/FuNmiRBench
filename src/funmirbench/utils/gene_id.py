"""Gene-ID detection heuristics for DE tables."""

from __future__ import annotations

import re
from typing import List

ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE)


def find_gene_id_column(df, *, threshold: float = 0.5) -> str:
    """
    Identify which column (or the index) holds gene identifiers.

    Returns the column name, or the sentinel ``"__index__"`` when gene IDs
    live in the DataFrame index.

    Priority:
    1. Explicit ``gene_id`` or ``gene_name`` column.
    2. The column whose values best match the Ensembl pattern (>= *threshold*).
    3. The index, if it matches the Ensembl pattern (>= *threshold*).

    Raises ``ValueError`` if no usable source is found.
    """
    for c in ("gene_id", "gene_name"):
        if c in df.columns:
            return c

    best_col = None
    best_frac = 0.0
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        frac = float(s.str.match(ENSEMBL_GENE_RE).mean())
        if frac > best_frac:
            best_frac = frac
            best_col = str(col)
    if best_col is not None and best_frac >= threshold:
        return best_col

    idx = df.index.astype(str)
    if len(idx) > 0:
        frac_idx = float(idx.str.match(ENSEMBL_GENE_RE).mean())
        if frac_idx >= threshold:
            return "__index__"

    raise ValueError(
        "Could not identify gene IDs in DE table. Expected a column named "
        "gene_id/gene_name, or Ensembl-like IDs (ENSG...) in a column or "
        "the index."
    )


def extract_gene_ids(df, *, threshold: float = 0.5) -> List[str]:
    """
    Return gene ID values as a list of strings.

    Uses :func:`find_gene_id_column` to locate the source, then extracts
    the values.
    """
    col = find_gene_id_column(df, threshold=threshold)
    if col == "__index__":
        return list(df.index.astype(str))
    return df[col].dropna().astype(str).tolist()
