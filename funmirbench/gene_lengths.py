"""Gene-length annotation helpers for manuscript plots."""

from __future__ import annotations

import gzip
import logging
import pathlib

import pandas as pd

from funmirbench.gene_ids import strip_ensembl_version
from funmirbench.protein_coding import (
    DEFAULT_GTF_REL_PATH,
    ENSEMBL_RELEASE,
    _parse_gtf_attributes,
    ensure_ensembl_gtf,
)

logger = logging.getLogger(__name__)

DEFAULT_UTR3_LENGTH_CACHE_REL_PATH = pathlib.Path(
    "data/resources/ensembl/utr3_lengths.tsv"
)
UTR3_FEATURE_NAMES = {
    "three_prime_utr",
    "three_prime_UTR",
    "3UTR",
    "3utr",
}


def _resolve_repo_path(root: pathlib.Path, path: str | pathlib.Path | None, default: pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(path) if path is not None else default
    if not out.is_absolute():
        out = root / out
    return out


def _interval_union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted((min(start, end), max(start, end)) for start, end in intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start + 1
        current_start, current_end = start, end
    total += current_end - current_start + 1
    return int(total)


def parse_utr3_lengths_from_gtf(
    gtf_gz_path: pathlib.Path,
    *,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Parse per-gene 3'UTR lengths from an Ensembl GTF.

    Lengths are computed as the union of all 3'UTR intervals observed for each
    Ensembl gene. This avoids double-counting overlapping 3'UTR segments across
    transcript isoforms while still representing the available 3'UTR territory
    for a gene.
    """
    intervals_by_gene: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(gtf_gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in UTR3_FEATURE_NAMES:
                continue
            attrs = _parse_gtf_attributes(fields[8])
            if protein_coding_only:
                biotype = (
                    attrs.get("gene_biotype")
                    or attrs.get("gene_type")
                    or attrs.get("transcript_biotype")
                    or attrs.get("transcript_type")
                )
                if biotype and biotype != "protein_coding":
                    continue
            gene_id = attrs.get("gene_id")
            if not gene_id:
                continue
            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                continue
            intervals_by_gene.setdefault(strip_ensembl_version(gene_id), []).append((start, end))

    rows = [
        {
            "gene_id": gene_id,
            "utr3_length_bp": _interval_union_length(intervals),
        }
        for gene_id, intervals in intervals_by_gene.items()
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No 3'UTR intervals parsed from {gtf_gz_path}")
    out = out.loc[out["utr3_length_bp"] > 0].sort_values("gene_id").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No positive 3'UTR lengths parsed from {gtf_gz_path}")
    return out


def read_utr3_length_cache(cache_path: pathlib.Path) -> pd.DataFrame:
    table = pd.read_csv(cache_path, sep="\t")
    required = {"gene_id", "utr3_length_bp"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{cache_path} missing required columns: {missing}")
    table = table[["gene_id", "utr3_length_bp"]].copy()
    table["gene_id"] = table["gene_id"].map(strip_ensembl_version)
    table["utr3_length_bp"] = pd.to_numeric(table["utr3_length_bp"], errors="coerce")
    table = table.dropna(subset=["gene_id", "utr3_length_bp"])
    table = table.loc[table["utr3_length_bp"] > 0].copy()
    return table.groupby("gene_id", as_index=False)["utr3_length_bp"].max()


def write_utr3_length_cache(cache_path: pathlib.Path, table: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table[["gene_id", "utr3_length_bp"]].to_csv(cache_path, sep="\t", index=False)


def load_utr3_lengths(
    *,
    root: pathlib.Path,
    gtf_path: str | pathlib.Path | None = None,
    cache_path: str | pathlib.Path | None = None,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Load cached Ensembl 3'UTR lengths, building them from the GTF if needed."""
    root = root.resolve()
    cache = _resolve_repo_path(root, cache_path, DEFAULT_UTR3_LENGTH_CACHE_REL_PATH)
    if cache.exists():
        table = read_utr3_length_cache(cache)
        if not table.empty:
            logger.info("Loaded %d 3'UTR lengths from %s", len(table), cache)
            return table

    gtf = ensure_ensembl_gtf(root=root, gtf_path=gtf_path or DEFAULT_GTF_REL_PATH)
    table = parse_utr3_lengths_from_gtf(gtf, protein_coding_only=protein_coding_only)
    write_utr3_length_cache(cache, table)
    logger.info(
        "Cached %d Ensembl release %s 3'UTR lengths to %s",
        len(table),
        ENSEMBL_RELEASE,
        cache,
    )
    return table
