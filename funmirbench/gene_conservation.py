"""3'UTR conservation annotation helpers for manuscript plots."""

from __future__ import annotations

import gzip
import logging
import pathlib

import numpy as np
import pandas as pd
import pyBigWig

from funmirbench.gene_ids import strip_ensembl_version
from funmirbench.gene_lengths import _interval_union_length, _merge_intervals
from funmirbench.protein_coding import (
    DEFAULT_GTF_REL_PATH,
    ENSEMBL_RELEASE,
    _download_file,
    _parse_gtf_attributes,
    ensure_ensembl_gtf,
)

logger = logging.getLogger(__name__)

DEFAULT_PHYLOP_URL = "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
DEFAULT_PHASTCONS_URL = "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw"
DEFAULT_PHYLOP_REL_PATH = pathlib.Path("data/resources/ucsc/hg38.phyloP100way.bw")
DEFAULT_PHASTCONS_REL_PATH = pathlib.Path("data/resources/ucsc/hg38.phastCons100way.bw")
DEFAULT_UTR3_CONSERVATION_CACHE_REL_PATH = pathlib.Path(
    "data/resources/ensembl/utr3_conservation_longest_protein_coding_transcript.tsv"
)
UTR3_FEATURE_NAMES = {"three_prime_utr", "three_prime_UTR", "3UTR", "3utr"}


def _resolve_repo_path(root: pathlib.Path, path: str | pathlib.Path | None, default: pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(path) if path is not None else default
    if not out.is_absolute():
        out = root / out
    return out


def _normalize_chrom_for_bigwig(chrom: str, chrom_sizes: dict[str, int]) -> str | None:
    chrom = str(chrom).strip()
    candidates = []
    if chrom == "MT":
        candidates.extend(["chrM", "MT", "M"])
    else:
        candidates.append(chrom)
        if chrom.startswith("chr"):
            candidates.append(chrom.removeprefix("chr"))
        else:
            candidates.append(f"chr{chrom}")
    for candidate in candidates:
        if candidate in chrom_sizes:
            return candidate
    return None


def ensure_conservation_bigwigs(
    *,
    root: pathlib.Path,
    phylop_path: str | pathlib.Path | None = None,
    phastcons_path: str | pathlib.Path | None = None,
    phylop_url: str = DEFAULT_PHYLOP_URL,
    phastcons_url: str = DEFAULT_PHASTCONS_URL,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Return local phyloP/phastCons BigWig paths, downloading defaults if needed."""
    root = root.resolve()
    phylop = _resolve_repo_path(root, phylop_path, DEFAULT_PHYLOP_REL_PATH)
    phastcons = _resolve_repo_path(root, phastcons_path, DEFAULT_PHASTCONS_REL_PATH)
    if not phylop.exists():
        logger.info("Downloading phyloP BigWig: %s", phylop)
        _download_file(phylop_url, phylop)
    if not phastcons.exists():
        logger.info("Downloading phastCons BigWig: %s", phastcons)
        _download_file(phastcons_url, phastcons)
    return phylop, phastcons


def parse_longest_protein_coding_utr3_transcript_intervals_from_gtf(
    gtf_gz_path: pathlib.Path,
    *,
    protein_coding_only: bool = True,
) -> dict[str, dict[str, object]]:
    """Parse the longest protein-coding 3'UTR transcript per gene.

    Returns a dictionary keyed by gene ID. Each value contains the selected
    transcript ID, merged intervals grouped by chromosome, selected transcript
    3'UTR length, and the number of protein-coding transcripts with annotated
    3'UTR intervals for that gene.
    """
    intervals_by_gene_tx: dict[str, dict[str, dict[str, list[tuple[int, int]]]]] = {}
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
                    attrs.get("transcript_biotype")
                    or attrs.get("transcript_type")
                    or attrs.get("gene_biotype")
                    or attrs.get("gene_type")
                )
                if biotype and biotype != "protein_coding":
                    continue
            gene_id = attrs.get("gene_id")
            transcript_id = attrs.get("transcript_id")
            if not gene_id or not transcript_id:
                continue
            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                continue
            gene_id = strip_ensembl_version(gene_id)
            transcript_id = strip_ensembl_version(transcript_id)
            chrom = fields[0]
            intervals_by_gene_tx.setdefault(gene_id, {}).setdefault(transcript_id, {}).setdefault(chrom, []).append((start, end))

    selected: dict[str, dict[str, object]] = {}
    for gene_id, transcript_intervals in intervals_by_gene_tx.items():
        candidates = []
        for transcript_id, chrom_intervals in transcript_intervals.items():
            merged_by_chrom = {
                chrom: _merge_intervals(intervals)
                for chrom, intervals in chrom_intervals.items()
                if intervals
            }
            length = sum(_interval_union_length(intervals) for intervals in merged_by_chrom.values())
            if length > 0:
                candidates.append((transcript_id, int(length), merged_by_chrom))
        if not candidates:
            continue
        transcript_id, length, merged_by_chrom = sorted(candidates, key=lambda item: (-item[1], item[0]))[0]
        selected[gene_id] = {
            "transcript_id": transcript_id,
            "utr3_length_bp": int(length),
            "utr3_interval_count": int(sum(len(intervals) for intervals in merged_by_chrom.values())),
            "protein_coding_transcript_with_utr3_count": int(len(candidates)),
            "intervals_by_chrom": merged_by_chrom,
        }
    if not selected:
        raise ValueError(f"No protein-coding transcript 3'UTR intervals parsed from {gtf_gz_path}")
    return selected


def _mean_bigwig_signal_for_transcript(
    bw,
    chrom_intervals: dict[str, list[tuple[int, int]]],
) -> tuple[float, int]:
    chrom_sizes = bw.chroms()
    chunks = []
    scored_bases = 0
    for chrom, intervals in chrom_intervals.items():
        bw_chrom = _normalize_chrom_for_bigwig(chrom, chrom_sizes)
        if bw_chrom is None:
            continue
        chrom_size = chrom_sizes[bw_chrom]
        for start_1based, end_1based in intervals:
            start_0based = max(0, int(start_1based) - 1)
            end_0based_exclusive = min(int(end_1based), chrom_size)
            if start_0based >= end_0based_exclusive:
                continue
            values = bw.values(bw_chrom, start_0based, end_0based_exclusive)
            arr = np.asarray([np.nan if value is None else value for value in values], dtype=float)
            valid = arr[np.isfinite(arr)]
            if valid.size:
                chunks.append(valid)
                scored_bases += int(valid.size)
    if not chunks:
        return np.nan, 0
    all_values = np.concatenate(chunks)
    return float(np.mean(all_values)), scored_bases


def build_utr3_conservation_table(
    *,
    gtf_gz_path: pathlib.Path,
    phylop_bw_path: pathlib.Path,
    phastcons_bw_path: pathlib.Path,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Compute mean 3'UTR conservation for each gene's longest protein-coding transcript."""
    selected_by_gene = parse_longest_protein_coding_utr3_transcript_intervals_from_gtf(
        gtf_gz_path,
        protein_coding_only=protein_coding_only,
    )
    rows = []
    with pyBigWig.open(str(phylop_bw_path)) as bw_phylop, pyBigWig.open(str(phastcons_bw_path)) as bw_phastcons:
        for gene_id, selected in selected_by_gene.items():
            chrom_intervals = selected["intervals_by_chrom"]
            mean_phylop, phylop_bases = _mean_bigwig_signal_for_transcript(bw_phylop, chrom_intervals)
            mean_phastcons, phastcons_bases = _mean_bigwig_signal_for_transcript(bw_phastcons, chrom_intervals)
            rows.append(
                {
                    "gene_id": gene_id,
                    "transcript_id": selected["transcript_id"],
                    "mean_phyloP": mean_phylop,
                    "mean_phastCons": mean_phastcons,
                    "utr3_scored_bases_phyloP": phylop_bases,
                    "utr3_scored_bases_phastCons": phastcons_bases,
                    "utr3_length_bp": int(selected["utr3_length_bp"]),
                    "utr3_interval_count": int(selected["utr3_interval_count"]),
                    "protein_coding_transcript_with_utr3_count": int(selected["protein_coding_transcript_with_utr3_count"]),
                }
            )
    out = pd.DataFrame(rows).sort_values("gene_id").reset_index(drop=True)
    if out.empty:
        raise ValueError("No per-gene conservation rows were created")
    return out


def read_utr3_conservation_cache(cache_path: pathlib.Path) -> pd.DataFrame:
    table = pd.read_csv(cache_path, sep="\t")
    required = {"gene_id", "transcript_id", "mean_phyloP"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{cache_path} missing required columns: {missing}")
    table = table.copy()
    table["gene_id"] = table["gene_id"].map(strip_ensembl_version)
    table["transcript_id"] = table["transcript_id"].map(strip_ensembl_version)
    for col in [
        "mean_phyloP",
        "mean_phastCons",
        "utr3_scored_bases_phyloP",
        "utr3_scored_bases_phastCons",
        "utr3_length_bp",
        "utr3_interval_count",
        "protein_coding_transcript_with_utr3_count",
    ]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table.sort_values(["gene_id", "utr3_length_bp"], ascending=[True, False]).drop_duplicates("gene_id")


def write_utr3_conservation_cache(cache_path: pathlib.Path, table: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, sep="\t", index=False)


def load_utr3_conservation(
    *,
    root: pathlib.Path,
    gtf_path: str | pathlib.Path | None = None,
    phylop_path: str | pathlib.Path | None = None,
    phastcons_path: str | pathlib.Path | None = None,
    cache_path: str | pathlib.Path | None = None,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Load cached 3'UTR conservation, building it from Ensembl GTF and UCSC BigWigs if needed."""
    root = root.resolve()
    cache = _resolve_repo_path(root, cache_path, DEFAULT_UTR3_CONSERVATION_CACHE_REL_PATH)
    if cache.exists():
        table = read_utr3_conservation_cache(cache)
        if not table.empty:
            logger.info("Loaded %d longest-transcript 3'UTR conservation rows from %s", len(table), cache)
            return table

    gtf = ensure_ensembl_gtf(root=root, gtf_path=gtf_path or DEFAULT_GTF_REL_PATH)
    phylop, phastcons = ensure_conservation_bigwigs(root=root, phylop_path=phylop_path, phastcons_path=phastcons_path)
    table = build_utr3_conservation_table(
        gtf_gz_path=gtf,
        phylop_bw_path=phylop,
        phastcons_bw_path=phastcons,
        protein_coding_only=protein_coding_only,
    )
    write_utr3_conservation_cache(cache, table)
    logger.info(
        "Cached %d Ensembl release %s longest-transcript 3'UTR conservation rows to %s",
        len(table),
        ENSEMBL_RELEASE,
        cache,
    )
    return table
