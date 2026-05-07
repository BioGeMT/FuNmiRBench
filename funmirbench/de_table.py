"""DE table reading and gene ID detection."""

import re
from pathlib import Path

import pandas as pd

_ENSEMBL = re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE)


def find_gene_id_column(df):
    """Identify which column (or index) holds gene identifiers."""
    for col in df.columns:
        low = str(col).lower().replace(" ", "_")
        if low in ("gene_id", "geneid", "gene_name", "genename"):
            return col

    best_col, best_frac = None, 0.0
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        frac = float(s.str.match(_ENSEMBL).mean())
        if frac > best_frac:
            best_frac = frac
            best_col = str(col)
    if best_col is not None and best_frac >= 0.5:
        return best_col

    idx = df.index.astype(str)
    if len(idx) > 0 and float(idx.str.match(_ENSEMBL).mean()) >= 0.5:
        return "__index__"

    return df.columns[0]


def extract_gene_ids(df):
    """Return gene ID values as a list of strings."""
    col = find_gene_id_column(df)
    if col == "__index__":
        return list(df.index.astype(str))
    return df[col].dropna().astype(str).tolist()


def read_de_table(path: Path) -> pd.DataFrame:
    """Read a DE table TSV, normalizing malformed gene ID headers."""
    df = pd.read_csv(path, sep="\t")
    cols = [str(c).strip() for c in df.columns]
    if cols[0] == "" or cols[0].startswith("Unnamed:"):
        cols[0] = "gene_id"
    df.columns = cols
    if (
        "gene_id" not in df.columns
        and len(df.index) > 0
        and not isinstance(df.index, pd.RangeIndex)
        and float(pd.Series(df.index.astype(str)).str.match(_ENSEMBL).mean()) >= 0.5
    ):
        df = df.reset_index().rename(columns={"index": "gene_id"})
    return df
