#!/usr/bin/env python3
"""
TargetScan v8 (vert_80) standardization pipeline (human).

What this script does
---------------------
1) Downloads the minimal TargetScan input files needed to reproduce the pipeline:
   - UTR_Sequences.txt
   - Summary_Counts.all_predictions.txt
   - miR_Family_Info.txt

2) Builds a "longest UTR transcript per gene" index from TargetScan UTR_Sequences for species=9606.

3) Downloads Ensembl v115 GTF (GRCh38) and builds cached mapping tables:
   - transcript_id (ENST stable) -> gene_id (ENSG stable)
   - gene_id (ENSG stable) -> gene_name

4) Runs QC overlap diagnostics against Ensembl v115 transcript IDs (after stripping version suffixes).

5) Downloads miRBase mature.fa pinned to release 22.1 (from archive URL) and parses accession->name mapping.

6) Builds two standardized prediction sets from Summary_Counts.all_predictions.txt:
   - targetscanCons     (filter: Total num conserved sites > 0; score: Cumulative weighted context++ score)
   - targetscanNonCons  (filter: Total num nonconserved sites > 0; score: Cumulative weighted context++ score)

Score handling
------------------------
- Keep rows even when the selected score column is NULL.
- NULL scores are imputed as the weakest observed non-NULL score for that predictor after mandatory filters.
- Write two score columns:
    Score      : raw direction-corrected score
    Score_norm : percentile rank in [0,1] used for benchmarking
- Score direction is standardized so that higher underlying score always means a stronger/more confident predicted interaction:
    - targetscanCons    : -1 * (Cumulative weighted context++ score)
    - targetscanNonCons : -1 * (Cumulative weighted context++ score)
- After direction correction and NULL imputation, scores are converted to within-predictor percentile ranks.
- Ranking is computed at the base-row level before family expansion, so miRNA family size does not affect ranks.

Outputs
-------
Standardized TSVs are written to:
  data/predictions/<predictor>/<predictor>_standardized.tsv

Schema:
  Ensembl_ID, Gene_Name, miRNA_ID, miRNA_Name, Score, Score_norm

Notes
-----
- TargetScan transcripts are versioned (ENST... .6). Ensembl mapping uses stable IDs,
  so to compare after stripping version.
"""

from __future__ import annotations

import csv
import gzip as gz
import logging
import math
import pathlib
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Global constants
# =============================================================================
MIRBASE_RELEASE = "22.1"
MIRBASE_MATURE_URL = "https://mirbase.org/download_version_files/22.1/mature.fa"


# =============================================================================
# Helpers
# =============================================================================
def _repo_root() -> pathlib.Path:
    # pipelines/standardized_predictors/targetscan/pipeline.py -> parents[3] is repo root
    return pathlib.Path(__file__).resolve().parents[3]


def _strip_version(x: str) -> str:
    """Return stable ID part before the first '.' (e.g. ENST... .5 -> ENST...)."""
    s = x.strip()
    return s.split(".", 1)[0] if "." in s else s


def _to_float_or_none(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "" or x.upper() == "NULL":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _utr_len(seq: str) -> int:
    """Count UTR length ignoring gaps and non-ACGTU chars."""
    s = (seq or "").strip().upper()
    return sum(1 for ch in s if ch in ("A", "C", "G", "U", "T"))


def _parse_gtf_attributes(attr_field: str) -> Dict[str, str]:
    """
    Parse a GTF attributes column into a dict.
    Example: gene_id "ENSG..."; transcript_id "ENST..."; gene_name "TP53";
    """
    out: Dict[str, str] = {}
    parts = [p.strip() for p in attr_field.strip().split(";") if p.strip()]
    for p in parts:
        if " " not in p:
            continue
        k, v = p.split(" ", 1)
        out[k] = v.strip().strip('"')
    return out


def _safe_unlink(p: pathlib.Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        return


def _download_url(url: str, dest: pathlib.Path) -> None:
    """Download URL to dest atomically via .part file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as out:
            shutil.copyfileobj(r, out)
        tmp.replace(dest)
    finally:
        _safe_unlink(tmp)


def _assert_fasta(path: pathlib.Path) -> None:
    """Guardrail: file must look like FASTA (first non-empty line starts with '>')."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith(">"):
                raise RuntimeError(f"Not FASTA (first line: {s[:120]})")
            return
    raise RuntimeError("FASTA check failed: file appears empty.")


def _directional_score(value: float, *, reverse: bool) -> float:
    """Return score with standardized direction: higher = stronger."""
    return -float(value) if reverse else float(value)


def _percentile_ranks(values: List[float]) -> List[float]:
    """
    Convert scores to percentile-like ranks in (0, 1], preserving order.
    Lowest score gets 1/N, highest score gets 1.0.
    Ties receive the average of their occupied ranks.
    """
    n = len(values)
    if n == 0:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1

        avg_rank = ((i + 1) + j) / 2.0
        pct = avg_rank / n

        for k in range(i, j):
            orig_idx = indexed[k][0]
            ranks[orig_idx] = pct

        i = j

    return ranks


# =============================================================================
# STEP 1/7 - Download + unzip TargetScan inputs
# =============================================================================
def step1_download_targetscan_files(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> Dict[str, pathlib.Path]:
    logger.info("\n=== STEP 1/7: Download + unzip TargetScan inputs (flat under predictor data/) ===")

    data_dir = pathlib.Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    base = "https://www.targetscan.org/vert_80/vert_80_data_download"
    zips: List[str] = [
        "UTR_Sequences.txt.zip",
        "Summary_Counts.all_predictions.txt.zip",
        "miR_Family_Info.txt.zip",
    ]

    def _download_if_needed(url: str, dest: pathlib.Path) -> None:
        if dest.exists() and not force:
            logger.info("Skipping %s (already exists)", dest.name)
            return
        logger.info("Downloading %s", url)
        _download_url(url, dest)
        logger.info("Downloaded %s", dest.name)

    def _unzip(zip_path: pathlib.Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                extracted = data_dir / member
                if extracted.exists() and not force:
                    continue
                zf.extract(member, data_dir)
        logger.info("Unzipped %s", zip_path.name)

    for fname in zips:
        url = f"{base}/{fname}"
        zip_path = data_dir / fname
        _download_if_needed(url, zip_path)
        if zip_path.exists():
            _unzip(zip_path)

    expected = {
        "UTR_Sequences.txt": data_dir / "UTR_Sequences.txt",
        "Summary_Counts.all_predictions.txt": data_dir / "Summary_Counts.all_predictions.txt",
        "miR_Family_Info.txt": data_dir / "miR_Family_Info.txt",
    }
    for k, p in expected.items():
        if not p.exists():
            raise FileNotFoundError(f"Expected file missing after download/unzip: {k} -> {p}")

    logger.info("✔ TargetScan inputs ready in %s", data_dir)
    return expected


# =============================================================================
# STEP 2/7 - Build longest-UTR transcript index
# =============================================================================
def step2_build_longest_utr_index(
    utr_sequences_path: pathlib.Path,
    *,
    species_id: str = "9606",
    report_top_n_tx_counts: int = 6,
) -> Dict[str, Any]:
    """
    Choose ONE transcript per gene: the transcript with the LONGEST annotated UTR,
    using TargetScan's UTR_Sequences.txt for a single species (default: human 9606).

    Returns:
      - best_tx_by_gene_id: TargetScan Gene ID -> (Transcript ID, UTR_len, Gene Symbol)
      - gene_id_by_tx: Transcript ID -> TargetScan Gene ID
      - gene_symbol_by_gene_id: TargetScan Gene ID -> Gene Symbol
    """
    logger.info("\n=== STEP 2/7: Build longest-UTR transcript index from UTR_Sequences.txt ===")

    utr_sequences_path = pathlib.Path(utr_sequences_path)
    if not utr_sequences_path.exists():
        raise FileNotFoundError(f"UTR sequences file not found: {utr_sequences_path}")

    tx_key = "Refseq ID"
    gene_id_key = "Gene ID"
    gene_sym_key = "Gene Symbol"
    sp_key = "Species ID"
    seq_key = "UTR sequence"

    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = {}
    gene_id_by_tx: Dict[str, str] = {}
    gene_symbol_by_gene_id: Dict[str, str] = {}

    tx_count_by_gene: Counter[str] = Counter()
    best_len_by_gene: Dict[str, int] = {}
    best_len_tie_count_by_gene: Counter[str] = Counter()

    n_total = 0
    n_species = 0

    with utr_sequences_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {utr_sequences_path}")

        required = [tx_key, gene_id_key, gene_sym_key, sp_key, seq_key]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Unexpected header in {utr_sequences_path} (missing {missing}). "
                f"Got: {reader.fieldnames}"
            )

        for row in reader:
            n_total += 1
            if row[sp_key].strip() != species_id:
                continue
            n_species += 1

            tx = row[tx_key].strip()
            gene_id = row[gene_id_key].strip()
            gene_sym = row[gene_sym_key].strip()
            L = _utr_len(row[seq_key])

            gene_id_by_tx[tx] = gene_id
            gene_symbol_by_gene_id[gene_id] = gene_sym
            tx_count_by_gene[gene_id] += 1

            prev_best_len = best_len_by_gene.get(gene_id)
            if prev_best_len is None:
                best_len_by_gene[gene_id] = L
                best_len_tie_count_by_gene[gene_id] = 1
            elif L > prev_best_len:
                best_len_by_gene[gene_id] = L
                best_len_tie_count_by_gene[gene_id] = 1
            elif L == prev_best_len:
                best_len_tie_count_by_gene[gene_id] += 1

            prev = best_tx_by_gene_id.get(gene_id)
            if prev is None or L > prev[1]:
                best_tx_by_gene_id[gene_id] = (tx, L, gene_sym)

    genes_with_utr = len(best_tx_by_gene_id)
    logger.info(
        "UTR_Sequences stats: rows_total=%d | rows_species(%s)=%d | genes_with_UTR=%d",
        n_total, species_id, n_species, genes_with_utr
    )

    dist = Counter(tx_count_by_gene.values())
    top_bins = dist.most_common(report_top_n_tx_counts)
    logger.info(
        "Transcripts-per-gene (top %d bins): %s",
        report_top_n_tx_counts,
        " | ".join([f"{k}tx:{v}" for k, v in top_bins]),
    )

    tie_genes = sum(1 for _g, c in best_len_tie_count_by_gene.items() if c >= 2)
    pct = (tie_genes / genes_with_utr) * 100.0 if genes_with_utr else 0.0
    logger.info(
        "Longest-UTR ties: genes_with_tie=%d (%.2f%%). Tie rule: keep first-seen transcript among tied max length.",
        tie_genes, pct
    )

    for gid, (tx, L, sym) in list(best_tx_by_gene_id.items())[:3]:
        logger.info("  sample longest UTR: %s -> %s (UTR_len=%d, symbol=%s)", gid, tx, L, sym)

    return {
        "best_tx_by_gene_id": best_tx_by_gene_id,
        "gene_id_by_tx": gene_id_by_tx,
        "gene_symbol_by_gene_id": gene_symbol_by_gene_id,
    }


# =============================================================================
# STEP 3/7 - Download Ensembl v115 GTF
# =============================================================================
def step3_download_ensembl115_gtf(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> pathlib.Path:
    logger.info("\n=== STEP 3/7: Download Ensembl v115 GTF (GRCh38) ===")

    data_dir = pathlib.Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/"
        "Homo_sapiens.GRCh38.115.gtf.gz"
    )
    dest = data_dir / "Homo_sapiens.GRCh38.115.gtf.gz"

    if dest.exists() and not force:
        logger.info("Skipping %s (already exists)", dest.name)
        return dest

    logger.info("Downloading %s", url)
    _download_url(url, dest)
    logger.info("Downloaded %s", dest.name)
    return dest


# =============================================================================
# STEP 4/7 - Build Ensembl mappings + cache to TSVs
# =============================================================================
def step4_build_and_cache_ensembl115_tables(
    ensembl_gtf_gz: pathlib.Path,
    *,
    cache_dir: pathlib.Path,
    force_rebuild: bool = False,
) -> Dict[str, Dict[str, str]]:
    logger.info("\n=== STEP 4/7: Build Ensembl v115 mapping tables + cache to TSVs ===")

    cache_dir = pathlib.Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tx2gene_path = cache_dir / "ensembl115_tx2gene.tsv.gz"
    gene2name_path = cache_dir / "ensembl115_gene2name.tsv.gz"

    def _load_cache() -> Optional[Dict[str, Dict[str, str]]]:
        if not tx2gene_path.exists() or not gene2name_path.exists():
            return None

        tx_to_gene: Dict[str, str] = {}
        gene_to_name: Dict[str, str] = {}

        with gz.open(tx2gene_path, "rt", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                tx_to_gene[row["transcript_id"]] = row["gene_id"]

        with gz.open(gene2name_path, "rt", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                gene_to_name[row["gene_id"]] = row["gene_name"]

        logger.info(
            "Loaded cached Ensembl tables: tx_to_gene=%d | gene_to_name=%d",
            len(tx_to_gene), len(gene_to_name)
        )
        return {"tx_to_gene": tx_to_gene, "gene_to_name": gene_to_name}

    if not force_rebuild:
        cached = _load_cache()
        if cached is not None:
            return cached

    ensembl_gtf_gz = pathlib.Path(ensembl_gtf_gz)
    if not ensembl_gtf_gz.exists():
        raise FileNotFoundError(f"Ensembl GTF not found: {ensembl_gtf_gz}")

    tx_to_gene: Dict[str, str] = {}
    gene_to_name: Dict[str, str] = {}

    n_total = 0
    n_transcript_rows = 0

    with gz.open(ensembl_gtf_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            n_total += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "transcript":
                continue
            n_transcript_rows += 1

            attrs = _parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id")
            tx_id = attrs.get("transcript_id")
            gene_name = attrs.get("gene_name")

            if not gene_id or not tx_id:
                continue

            gene_id_s = _strip_version(gene_id)
            tx_id_s = _strip_version(tx_id)

            tx_to_gene[tx_id_s] = gene_id_s
            if gene_name:
                gene_to_name.setdefault(gene_id_s, gene_name)

    logger.info(
        "Ensembl GTF parsed: lines=%d | transcript_features=%d | tx_mapped=%d | genes_named=%d",
        n_total, n_transcript_rows, len(tx_to_gene), len(gene_to_name)
    )

    with gz.open(tx2gene_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["transcript_id", "gene_id"], delimiter="\t")
        w.writeheader()
        for tx_id, gene_id in tx_to_gene.items():
            w.writerow({"transcript_id": tx_id, "gene_id": gene_id})

    with gz.open(gene2name_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gene_id", "gene_name"], delimiter="\t")
        w.writeheader()
        for gene_id, gene_name in gene_to_name.items():
            w.writerow({"gene_id": gene_id, "gene_name": gene_name})

    logger.info("Wrote cached Ensembl tables:")
    logger.info("  - %s", tx2gene_path)
    logger.info("  - %s", gene2name_path)

    return {"tx_to_gene": tx_to_gene, "gene_to_name": gene_to_name}


# =============================================================================
# STEP 5/7 - QC overlap TargetScan vs Ensembl transcripts
# =============================================================================
def step5_qc_targetscan_vs_ensembl_transcripts(
    utr_index: Dict[str, Any],
    ensembl_tables: Dict[str, Dict[str, str]],
    *,
    report_n: int = 10,
) -> None:
    logger.info("\n=== STEP 5/7: QC TargetScan transcript overlap with Ensembl v115 ===")

    gene_id_by_tx: Dict[str, str] = utr_index["gene_id_by_tx"]
    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = utr_index["best_tx_by_gene_id"]
    tx_to_gene_ensembl = ensembl_tables["tx_to_gene"]

    ts_all_tx = list(gene_id_by_tx.keys())
    ts_all_tx_stripped = [_strip_version(t) for t in ts_all_tx]

    exact_hits = sum(1 for t in ts_all_tx if _strip_version(t) in tx_to_gene_ensembl)
    stripped_hits = sum(1 for t in ts_all_tx_stripped if t in tx_to_gene_ensembl)

    logger.info("TargetScan UTR transcripts total: %d", len(ts_all_tx))
    logger.info("Ensembl v115 transcripts indexed: %d", len(tx_to_gene_ensembl))
    logger.info("Overlap (exact transcript_id after strip): %d", exact_hits)
    logger.info("Overlap (strip version before matching): %d", stripped_hits)

    ts_longest_tx = [tx for (tx, _L, _sym) in best_tx_by_gene_id.values()]
    ts_longest_tx_stripped = [_strip_version(t) for t in ts_longest_tx]
    longest_hits = sum(1 for t in ts_longest_tx_stripped if t in tx_to_gene_ensembl)

    logger.info("TargetScan longest-UTR transcripts: %d", len(ts_longest_tx))
    logger.info("Longest-UTR overlap with Ensembl (strip version): %d", longest_hits)

    misses = []
    for raw, stripped in zip(ts_longest_tx, ts_longest_tx_stripped):
        if stripped not in tx_to_gene_ensembl:
            misses.append((raw, stripped))
            if len(misses) >= report_n:
                break
    if misses:
        logger.info("Sample longest-UTR transcript misses (raw -> stripped):")
        for raw, stripped in misses:
            logger.info("  %s -> %s", raw, stripped)

    non_enst = [tx for tx in ts_longest_tx_stripped if not tx.startswith("ENST")]
    if non_enst:
        logger.info("Longest-UTR transcripts not ENST*: %d (cannot map via Ensembl transcript IDs)", len(non_enst))
        for tx in non_enst[: min(report_n, len(non_enst))]:
            logger.info("  non-ENST example: %s", tx)


# =============================================================================
# STEP 6/7 - Download miRBase mature.fa (pinned) + parse
# =============================================================================
def step6_download_mirbase_mature(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> pathlib.Path:
    logger.info("\n=== STEP 6/7: Download miRBase mature.fa (pinned release %s) ===", MIRBASE_RELEASE)
    data_dir = pathlib.Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / f"mirbase_mature_v{MIRBASE_RELEASE}.fa"
    if dest.exists() and not force:
        logger.info("Skipping miRBase mature.fa (already exists): %s", dest)
        return dest

    logger.info("Downloading %s", MIRBASE_MATURE_URL)
    _download_url(MIRBASE_MATURE_URL, dest)
    _assert_fasta(dest)

    logger.info("Downloaded miRBase mature.fa -> %s", dest)
    return dest


def parse_mirbase_mature(mature_fa: pathlib.Path) -> Dict[str, str]:
    """
    Parse mature.fa header lines:
      >hsa-miR-21-5p MIMAT0000076 Homo sapiens miR-21-5p
    Returns: accession -> miRNA name
    """
    logger.info("Parsing miRBase mature.fa")
    acc2name: Dict[str, str] = {}
    with pathlib.Path(mature_fa).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    mir_name = parts[0]
                    mir_acc = parts[1]
                    acc2name[mir_acc] = mir_name
    logger.info("Parsed %d mature miRNAs from miRBase", len(acc2name))
    return acc2name


# =============================================================================
# miR family -> human mature miRNAs (many-to-many expansion, validated to miRBase v22.1)
# =============================================================================
@dataclass(frozen=True)
class MirnaEntry:
    mirna_id: str    # miRBase Accession (MIMAT...)
    mirna_name: str  # miRBase ID (e.g. hsa-miR-183-5p)


def step_mirfamily_to_human_matures(
    mir_family_info_path: pathlib.Path,
    *,
    mirbase_acc2name: Dict[str, str],
    species_id: str = "9606",
) -> Dict[str, List[MirnaEntry]]:
    logger.info("\n=== STEP 7/7 Build miRNA family -> human mature miRNAs mapping (validated against miRBase v22.1) ===")

    p = pathlib.Path(mir_family_info_path)
    if not p.exists():
        raise FileNotFoundError(f"miR family file not found: {p}")

    fam2mirs: Dict[str, List[MirnaEntry]] = defaultdict(list)
    seen: Dict[str, set[Tuple[str, str]]] = defaultdict(set)

    n_total = 0
    n_species = 0
    n_missing_fields = 0
    n_acc_missing_in_mirbase = 0
    n_name_match = 0
    n_name_replaced = 0
    n_kept = 0

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {p}")

        required = ["Seed+m8", "Species ID", "MiRBase ID", "MiRBase Accession"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {p}")

        for row in reader:
            n_total += 1
            if row["Species ID"].strip() != species_id:
                continue
            n_species += 1

            seed = row["Seed+m8"].strip()
            mir_name_ts = row["MiRBase ID"].strip()
            mir_acc = row["MiRBase Accession"].strip()

            if not seed or not mir_name_ts or not mir_acc:
                n_missing_fields += 1
                continue

            mir_name_v22 = mirbase_acc2name.get(mir_acc)
            if mir_name_v22 is None:
                n_acc_missing_in_mirbase += 1
                continue

            if mir_name_ts == mir_name_v22:
                n_name_match += 1
            else:
                n_name_replaced += 1

            pair = (mir_acc, mir_name_v22)
            if pair in seen[seed]:
                continue
            seen[seed].add(pair)

            fam2mirs[seed].append(MirnaEntry(mirna_id=mir_acc, mirna_name=mir_name_v22))
            n_kept += 1

    logger.info(
        "miR_Family_Info stats: rows_total=%d | rows_species(%s)=%d | kept=%d | unique_families=%d",
        n_total, species_id, n_species, n_kept, len(fam2mirs)
    )
    logger.info(
        "miRNA annotation QC vs miRBase v%s: name_match=%d | name_replaced=%d | dropped_missing_accession=%d | dropped_missing_fields=%d",
        MIRBASE_RELEASE, n_name_match, n_name_replaced, n_acc_missing_in_mirbase, n_missing_fields
    )

    if fam2mirs:
        k = next(iter(fam2mirs.keys()))
        logger.info(
            "  sample family: %s -> %d human miRNAs after miRBase validation (e.g., %s)",
            k, len(fam2mirs[k]), fam2mirs[k][0].mirna_name
        )

    return dict(fam2mirs)


# =============================================================================
# Write standardized predictor file with direction-corrected percentile rank score
# =============================================================================
def step_write_standardized_predictions(
    summary_counts_path: pathlib.Path,
    *,
    utr_index: Dict[str, Any],
    ensembl_tables: Dict[str, Dict[str, str]],
    family_to_mirs: Dict[str, List[MirnaEntry]],
    out_predictions_dir: pathlib.Path,
    species_id: str = "9606",
) -> None:
    logger.info("\n=== Write standardized predictions (targetscanCNN/Cons/NonCons) ===")

    summary_counts_path = pathlib.Path(summary_counts_path)
    if not summary_counts_path.exists():
        raise FileNotFoundError(f"Summary counts file not found: {summary_counts_path}")

    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = utr_index["best_tx_by_gene_id"]
    gene_id_by_tx_ts: Dict[str, str] = utr_index["gene_id_by_tx"]

    tx_to_gene_ens = ensembl_tables["tx_to_gene"]
    gene_to_name_ens = ensembl_tables["gene_to_name"]

    out_predictions_dir = pathlib.Path(out_predictions_dir)
    out_predictions_dir.mkdir(parents=True, exist_ok=True)

    SCORE_SPECS = {
        "targetscanCons": {
            "score_col": "Cumulative weighted context++ score",
            "reverse": True,
        },
        "targetscanNonCons": {
            "score_col": "Cumulative weighted context++ score",
            "reverse": True,
        },
    }

    required_common = [
        "Transcript ID",
        "Gene Symbol",
        "miRNA family",
        "Species ID",
        "Total num conserved sites",
        "Total num nonconserved sites",
    ]
    required_scores = sorted({spec["score_col"] for spec in SCORE_SPECS.values()})

    # -------------------------------------------------------------------------
    # Pass 1: compute weakest observed direction-corrected score per predictor
    # -------------------------------------------------------------------------
    stats = {pred: Counter() for pred in SCORE_SPECS}
    weakest_nonnull = {pred: math.inf for pred in SCORE_SPECS}
    strongest_nonnull = {pred: -math.inf for pred in SCORE_SPECS}

    with summary_counts_path.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {summary_counts_path}")

        missing = [c for c in required_common if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {summary_counts_path}")

        missing_scores = [c for c in required_scores if c not in reader.fieldnames]
        if missing_scores:
            raise ValueError(f"Missing score columns {missing_scores} in {summary_counts_path}")

        for row in reader:
            if row["Species ID"].strip() != species_id:
                continue

            tx_raw = row["Transcript ID"].strip()
            gene_id_ts = gene_id_by_tx_ts.get(tx_raw)
            if gene_id_ts is None:
                continue

            best = best_tx_by_gene_id.get(gene_id_ts)
            if best is None or tx_raw != best[0]:
                continue

            tx_stable = _strip_version(tx_raw)
            gene_id_ens = tx_to_gene_ens.get(tx_stable)
            if gene_id_ens is None:
                continue

            fam = row["miRNA family"].strip()
            mirs = family_to_mirs.get(fam)
            if not mirs:
                continue

            try:
                n_cons = int(float(row["Total num conserved sites"]))
            except Exception:
                n_cons = 0
            try:
                n_noncons = int(float(row["Total num nonconserved sites"]))
            except Exception:
                n_noncons = 0

            memberships = {
                "targetscanCNN": True,
                "targetscanCons": n_cons > 0,
                "targetscanNonCons": n_noncons > 0,
            }

            for pred, spec in SCORE_SPECS.items():
                if not memberships[pred]:
                    continue

                stats[pred]["rows_after_filters"] += 1
                v = _to_float_or_none(row[spec["score_col"]])
                if v is None:
                    stats[pred]["rows_null_score"] += 1
                    continue

                score = _directional_score(v, reverse=spec["reverse"])
                stats[pred]["rows_nonnull_score"] += 1

                if score < weakest_nonnull[pred]:
                    weakest_nonnull[pred] = score
                if score > strongest_nonnull[pred]:
                    strongest_nonnull[pred] = score

    # Decide NULL imputation value in direction-corrected score space
    null_fill = {}
    for pred in SCORE_SPECS:
        if stats[pred]["rows_after_filters"] == 0:
            raise ValueError(f"No rows remain after filters for {pred}.")
        if stats[pred]["rows_nonnull_score"] == 0:
            raise ValueError(f"No non-NULL scores remain after filters for {pred}; cannot impute NULLs.")

        null_fill[pred] = float(weakest_nonnull[pred])

        logger.info(
            "%s score column '%s': weakest(non-NULL)=%.6g strongest(non-NULL)=%.6g | NULL rows=%d (imputed as %.6g)",
            pred,
            SCORE_SPECS[pred]["score_col"],
            weakest_nonnull[pred],
            strongest_nonnull[pred],
            stats[pred]["rows_null_score"],
            null_fill[pred],
        )

    # -------------------------------------------------------------------------
    # Pass 2: collect base rows with direction-corrected scores
    # -------------------------------------------------------------------------
    rows_by_pred: Dict[str, List[Dict[str, Any]]] = {pred: [] for pred in SCORE_SPECS}

    with summary_counts_path.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        assert reader.fieldnames is not None

        for row in reader:
            if row["Species ID"].strip() != species_id:
                continue

            tx_raw = row["Transcript ID"].strip()
            gene_id_ts = gene_id_by_tx_ts.get(tx_raw)
            if gene_id_ts is None:
                continue

            best = best_tx_by_gene_id.get(gene_id_ts)
            if best is None or tx_raw != best[0]:
                continue

            tx_stable = _strip_version(tx_raw)
            gene_id_ens = tx_to_gene_ens.get(tx_stable)
            if gene_id_ens is None:
                continue

            gene_name = gene_to_name_ens.get(gene_id_ens) or row["Gene Symbol"].strip()

            fam = row["miRNA family"].strip()
            mirs = family_to_mirs.get(fam)
            if not mirs:
                continue

            try:
                n_cons = int(float(row["Total num conserved sites"]))
            except Exception:
                n_cons = 0
            try:
                n_noncons = int(float(row["Total num nonconserved sites"]))
            except Exception:
                n_noncons = 0

            memberships = {
                "targetscanCNN": True,
                "targetscanCons": n_cons > 0,
                "targetscanNonCons": n_noncons > 0,
            }

            for pred, spec in SCORE_SPECS.items():
                if not memberships[pred]:
                    continue

                v = _to_float_or_none(row[spec["score_col"]])
                if v is None:
                    score = null_fill[pred]
                else:
                    score = _directional_score(v, reverse=spec["reverse"])

                rows_by_pred[pred].append(
                    {
                        "Ensembl_ID": gene_id_ens,
                        "Gene_Name": gene_name,
                        "mirs": mirs,
                        "score": float(score),
                    }
                )

    # -------------------------------------------------------------------------
    # Pass 3: percentile-rank normalize per predictor and write outputs
    # -------------------------------------------------------------------------
    writers: Dict[str, Tuple[csv.DictWriter, Any]] = {}
    out_paths: Dict[str, pathlib.Path] = {}
    written = {pred: Counter() for pred in SCORE_SPECS}

    def _open_writer(pred: str) -> Tuple[csv.DictWriter, Any, pathlib.Path]:
        out_dir = out_predictions_dir / pred
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pred}_standardized.tsv"
        fout = out_path.open("w", encoding="utf-8", newline="")
        w = csv.DictWriter(
            fout,
            fieldnames=["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score", "Score_norm"],
            delimiter="\t",
        )
        w.writeheader()
        return w, fout, out_path

    for pred in SCORE_SPECS:
        w, fout, p = _open_writer(pred)
        writers[pred] = (w, fout)
        out_paths[pred] = p

    for pred in SCORE_SPECS:
        rows = rows_by_pred[pred]
        scores = [r["score"] for r in rows]
        rank_scores = _percentile_ranks(scores)

        for row_obj, rank_score in zip(rows, rank_scores):
            for mir in row_obj["mirs"]:
                writers[pred][0].writerow(
                    {
                        "Ensembl_ID": row_obj["Ensembl_ID"],
                        "Gene_Name": row_obj["Gene_Name"],
                        "miRNA_ID": mir.mirna_id,
                        "miRNA_Name": mir.mirna_name,
                        "Score": f"{float(row_obj['score']):.6g}",
                        "Score_norm": f"{float(rank_score):.6g}",
                    }
                )
                written[pred]["written_rows"] += 1

            written[pred]["base_rows"] += 1

        logger.info(
            "%s rank normalization: base_rows=%d | min_rank=%.6g | max_rank=%.6g",
            pred,
            len(rows),
            min(rank_scores) if rank_scores else float("nan"),
            max(rank_scores) if rank_scores else float("nan"),
        )

    for pred in SCORE_SPECS:
        writers[pred][1].close()
        logger.info(
            "%s -> wrote %d rows (family-expanded) from %d base rows. output=%s",
            pred,
            written[pred]["written_rows"],
            written[pred]["base_rows"],
            out_paths[pred],
        )


def compute_final_statistics(predictions_root: pathlib.Path) -> None:
    logger.info("\n=== FINAL STATISTICS ===")

    sets = ["targetscanCons", "targetscanNonCons"]
    genes_by_set = {}
    mirs_by_set = {}

    for s in sets:
        p = predictions_root / s / f"{s}_standardized.tsv"
        genes = set()
        mirs = set()
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                genes.add(row["Ensembl_ID"])
                mirs.add(row["miRNA_ID"])
        genes_by_set[s] = genes
        mirs_by_set[s] = mirs

        logger.info("%s: %d unique genes | %d unique miRNAs", s, len(genes), len(mirs))

    common_genes = set.intersection(*genes_by_set.values())
    common_mirs = set.intersection(*mirs_by_set.values())

    logger.info("Overlap (both sets): %d genes | %d miRNAs", len(common_genes), len(common_mirs))


def main() -> None:
    repo_root = _repo_root()

    targetscan_dir = repo_root / "pipelines" / "standardized_predictors" / "targetscan"
    data_dir = targetscan_dir / "data"

    out_predictions_dir = repo_root / "data" / "predictions"

    files = step1_download_targetscan_files(data_dir, force=False)

    utr_index = step2_build_longest_utr_index(files["UTR_Sequences.txt"], species_id="9606")

    ensembl_gtf = step3_download_ensembl115_gtf(data_dir, force=False)
    ensembl_tables = step4_build_and_cache_ensembl115_tables(
        ensembl_gtf,
        cache_dir=data_dir,
        force_rebuild=False,
    )

    step5_qc_targetscan_vs_ensembl_transcripts(
        utr_index=utr_index,
        ensembl_tables=ensembl_tables,
    )

    mirbase_fa = step6_download_mirbase_mature(data_dir, force=False)
    mirbase_acc2name = parse_mirbase_mature(mirbase_fa)

    family_to_mirs = step_mirfamily_to_human_matures(
        files["miR_Family_Info.txt"],
        mirbase_acc2name=mirbase_acc2name,
        species_id="9606",
    )

    step_write_standardized_predictions(
        files["Summary_Counts.all_predictions.txt"],
        utr_index=utr_index,
        ensembl_tables=ensembl_tables,
        family_to_mirs=family_to_mirs,
        out_predictions_dir=out_predictions_dir,
        species_id="9606",
    )

    compute_final_statistics(out_predictions_dir)


if __name__ == "__main__":
    main()