#!/usr/bin/env python3
"""
TargetScan v8 (vert_80) standardization pipeline (human).

What this script does (current scope)
-------------------------------------
1) Downloads the minimal TargetScan input files needed to reproduce the pipeline:
   - UTR_Sequences.txt          (to choose a single transcript per gene, using TargetScan's transcript universe)
   - Summary_Counts.all_predictions.txt
   - miR_Family_Info.txt        (to expand miRNA families to human mature miRNAs)

2) Builds a "longest UTR transcript per gene" index **from TargetScan UTR_Sequences** for species=9606.
   Rationale: TargetScan outputs are transcript-level, and we need a deterministic gene-level representation
   that is consistent with the transcript universe used by TargetScan.

3) Downloads Ensembl v115 GTF (GRCh38) and builds cached mapping tables:
   - transcript_id (ENST, stable) -> gene_id (ENSG, stable)
   - gene_id (ENSG, stable) -> gene_name
   This supports normalization to Ensembl v115.

4) Runs QC overlap diagnostics: how many TargetScan transcripts (and chosen longest-UTR transcripts)
   match Ensembl v115 transcript IDs (after stripping version suffixes).

5) Builds *three* standardized prediction sets from Summary_Counts.all_predictions.txt:
   - targetscanCNN      (all rows with a non-NULL occupancy score)
   - targetscanCons     (subset: Total num conserved sites > 0)
   - targetscanNonCons  (subset: Total num nonconserved sites > 0)

   Each row is expanded family->human mature miRNAs (many-to-many expansion).
   Score is min-max normalized to [0,1] **after all mandatory filters** for the CNN set.
   NULL score rows are dropped (TargetScan did not report an occupancy score for that row).

Outputs
-------
Standardized TSVs are written to (repo-root):
  data/predictions/<predictor>/<predictor>_standardized.tsv

Schema (required + score):
  Ensembl_ID, Gene_Name, miRNA_ID, miRNA_Name, Score

Notes
-----
- Transcript matching: TargetScan uses versioned ENST IDs (ENST... .6). Ensembl mapping uses stable IDs,
  so we compare after stripping version (ENST000...).
- Some TargetScan "transcripts" (e.g., CDR1as) are not Ensembl transcripts; these are quantified and dropped
  at the Ensembl-mapping stage (since Ensembl_ID is mandatory).
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
from typing import Any, Dict, Iterable, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


# =============================================================================
# STEP 1/6 — Download + unzip TargetScan inputs (flat under predictor data/)
# =============================================================================
def step1_download_targetscan_files(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> Dict[str, pathlib.Path]:
    """
    Minimal TargetScan v8.0 inputs:
      - UTR_Sequences.txt.zip              (choose longest UTR transcript per gene)
      - Summary_Counts.all_predictions...  (contains scores + conserved/nonconserved counts)
      - miR_Family_Info.txt.zip            (map family -> miRNA IDs/strings)

    Stored FLAT under: pipelines/standardized_predictors/targetscan/data/
    """
    logger.info("\n=== STEP 1/6: Download + unzip TargetScan inputs (flat under predictor data/) ===")

    data_dir = pathlib.Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    base = "https://www.targetscan.org/vert_80/vert_80_data_download"
    zips: List[str] = [
        "UTR_Sequences.txt.zip",
        "Summary_Counts.all_predictions.txt.zip",
        "miR_Family_Info.txt.zip",
    ]

    def _download(url: str, dest: pathlib.Path) -> None:
        if dest.exists() and not force:
            logger.info("Skipping %s (already exists)", dest.name)
            return
        logger.info("Downloading %s", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url) as response, tmp.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
            tmp.replace(dest)
            logger.info("Downloaded %s", dest.name)
        finally:
            _safe_unlink(tmp)

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
        _download(url, zip_path)
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
# STEP 2/6 — Build longest-UTR transcript index (TargetScan-consistent)
# =============================================================================
def step2_build_longest_utr_index(
    utr_sequences_path: pathlib.Path,
    *,
    species_id: str = "9606",
    report_top_n_tx_counts: int = 6,
) -> Dict[str, Any]:
    """
    Choose ONE transcript per gene: the transcript with the LONGEST annotated UTR.

    IMPORTANT: UTR length is computed from TargetScan's own UTR_Sequences.txt, keeping the
    selection consistent with TargetScan's transcript universe and alignments.

    Returns:
      - best_tx_by_gene_id: TargetScan Gene ID -> (Transcript ID, UTR_len, Gene Symbol)
      - gene_id_by_tx: Transcript ID -> TargetScan Gene ID
      - gene_symbol_by_gene_id: TargetScan Gene ID -> Gene Symbol
    """
    logger.info("\n=== STEP 2/6: Build longest-UTR transcript index from UTR_Sequences.txt ===")

    utr_sequences_path = pathlib.Path(utr_sequences_path)
    if not utr_sequences_path.exists():
        raise FileNotFoundError(f"UTR sequences file not found: {utr_sequences_path}")

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

        # TargetScan uses "Refseq ID" (as in your head output). Keep it flexible.
        tx_key = None
        for candidate in ("Transcript ID", "Refseq ID", "RefSeq ID"):
            if candidate in reader.fieldnames:
                tx_key = candidate
                break
        if tx_key is None:
            raise ValueError(f"Cannot find Transcript/Refseq column in header: {reader.fieldnames}")

        gene_id_key = "Gene ID"
        gene_sym_key = "Gene Symbol"
        sp_key = "Species ID"

        seq_key = None
        for candidate in ("UTR Sequence", "UTR sequence", "UTR_sequence", "UTR"):
            if candidate in reader.fieldnames:
                seq_key = candidate
                break
        if seq_key is None:
            raise ValueError(f"Cannot find UTR sequence column in header: {reader.fieldnames}")

        missing = [c for c in (gene_id_key, gene_sym_key, sp_key) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {utr_sequences_path}")

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
    small_bins = [1, 2, 3, 4, 5]
    small_parts = [f"{k}tx:{dist[k]}" for k in small_bins if k in dist]
    if small_parts:
        logger.info("Transcripts-per-gene (common bins): %s", " | ".join(small_parts))

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
# STEP 3/6 — Download Ensembl v115 GTF
# =============================================================================
def step3_download_ensembl115_gtf(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> pathlib.Path:
    """Download Ensembl release 115 human GTF (GRCh38) into predictor data/."""
    logger.info("\n=== STEP 3/6: Download Ensembl v115 GTF (GRCh38) ===")

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
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(dest)
        logger.info("Downloaded %s", dest.name)
    finally:
        _safe_unlink(tmp)

    return dest


# =============================================================================
# STEP 4/6 — Build Ensembl mappings + write cached TSVs (hybrid approach)
# =============================================================================
def step4_build_and_cache_ensembl115_tables(
    ensembl_gtf_gz: pathlib.Path,
    *,
    cache_dir: pathlib.Path,
    force_rebuild: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Hybrid approach:
      - Parse Ensembl v115 GTF ONCE
      - Write lightweight cached TSVs for fast re-runs:
          ensembl115_tx2gene.tsv.gz
          ensembl115_gene2name.tsv.gz
      - On subsequent runs, load cached TSVs unless force_rebuild=True
    """
    logger.info("\n=== STEP 4/6: Build Ensembl v115 mapping tables + cache to TSVs ===")

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
# STEP 5/6 — QC overlap TargetScan vs Ensembl transcripts (+ non-Ensembl IDs)
# =============================================================================
def step5_qc_targetscan_vs_ensembl_transcripts(
    utr_index: Dict[str, Any],
    ensembl_tables: Dict[str, Dict[str, str]],
    *,
    report_n: int = 10,
) -> None:
    logger.info("\n=== STEP 5/6: QC TargetScan transcript overlap with Ensembl v115 ===")

    gene_id_by_tx: Dict[str, str] = utr_index["gene_id_by_tx"]
    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = utr_index["best_tx_by_gene_id"]
    tx_to_gene_ensembl = ensembl_tables["tx_to_gene"]

    ts_all_tx = list(gene_id_by_tx.keys())
    ts_all_tx_stripped = [_strip_version(t) for t in ts_all_tx]

    exact_hits = sum(1 for t in ts_all_tx if t in tx_to_gene_ensembl)
    stripped_hits = sum(1 for t in ts_all_tx_stripped if t in tx_to_gene_ensembl)

    logger.info("TargetScan UTR transcripts total: %d", len(ts_all_tx))
    logger.info("Ensembl v115 transcripts indexed: %d", len(tx_to_gene_ensembl))
    logger.info("Overlap (exact transcript_id): %d", exact_hits)
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

    # Non-Ensembl-like IDs (e.g., CDR1as): count how many longest-UTR tx are not ENST...
    non_enst = [tx for tx in ts_longest_tx_stripped if not tx.startswith("ENST")]
    if non_enst:
        logger.info("Longest-UTR transcripts not ENST*: %d (these cannot map via Ensembl transcript IDs)", len(non_enst))
        for tx in non_enst[: min(report_n, len(non_enst))]:
            logger.info("  non-ENST example: %s", tx)


# =============================================================================
# miR family -> human mature miRNAs (many-to-many expansion)
# =============================================================================
@dataclass(frozen=True)
class MirnaEntry:
    mirna_id: str    # miRBase Accession (MIMAT...)
    mirna_name: str  # MiRBase ID (e.g. hsa-miR-183-5p.2)


def step_mirfamily_to_human_matures(
    mir_family_info_path: pathlib.Path,
    *,
    species_id: str = "9606",
) -> Dict[str, List[MirnaEntry]]:
    """
    Build family -> list of (miRNA_ID, miRNA_Name) for the requested species.

    TargetScan 'miRNA family' values in Summary_Counts are seed strings (e.g., GAGGUAG).
    In miR_Family_Info, those appear as 'Seed+m8'. We map:
        Seed+m8 (string) -> human miRNAs (MiRBase Accession + MiRBase ID)

    We keep ALL human mature miRNAs for a family (many-to-many expansion).
    """
    logger.info("\n=== EXTRA: Build miRNA family -> human mature miRNAs mapping ===")

    p = pathlib.Path(mir_family_info_path)
    if not p.exists():
        raise FileNotFoundError(f"miR family file not found: {p}")

    fam2mirs: Dict[str, List[MirnaEntry]] = defaultdict(list)

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {p}")

        required = ["Seed+m8", "Species ID", "MiRBase ID", "MiRBase Accession"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {p}")

        n_total = 0
        n_species = 0
        n_kept = 0

        for row in reader:
            n_total += 1
            if row["Species ID"].strip() != species_id:
                continue
            n_species += 1

            seed = row["Seed+m8"].strip()
            mir_name = row["MiRBase ID"].strip()
            mir_acc = row["MiRBase Accession"].strip()

            if not seed or not mir_name or not mir_acc:
                continue

            fam2mirs[seed].append(MirnaEntry(mirna_id=mir_acc, mirna_name=mir_name))
            n_kept += 1

    logger.info(
        "miR_Family_Info stats: rows_total=%d | rows_species(%s)=%d | mapped_entries=%d | unique_families=%d",
        n_total, species_id, n_species, n_kept, len(fam2mirs)
    )

    # Small sanity check output
    if fam2mirs:
        k = next(iter(fam2mirs.keys()))
        logger.info("  sample family: %s -> %d human miRNAs (e.g., %s)",
                    k, len(fam2mirs[k]), fam2mirs[k][0].mirna_name)

    return dict(fam2mirs)


# =============================================================================
# STEP 6/6 — Build standardized predictor files (3 splits)
# =============================================================================
def step6_write_standardized_predictions(
    summary_counts_path: pathlib.Path,
    *,
    utr_index: Dict[str, Any],
    ensembl_tables: Dict[str, Dict[str, str]],
    family_to_mirs: Dict[str, List[MirnaEntry]],
    out_predictions_dir: pathlib.Path,
    species_id: str = "9606",
    score_col: str = "Predicted occupancy - transfected miRNA",
) -> None:
    """
    Write three standardized predictor TSVs under:
      data/predictions/<predictor>/<predictor>_standardized.tsv

    Predictor splits:
      - targetscanCNN: all rows with non-NULL score
      - targetscanCons: Total num conserved sites > 0
      - targetscanNonCons: Total num nonconserved sites > 0

    Normalization:
      - min/max computed from the FINAL filtered CNN rows (after Ensembl mapping + family mapping + longest UTR).
      - Score_norm = (Score_raw - min) / (max - min), clamped to [0,1].
    """
    logger.info("\n=== STEP 6/6: Write standardized predictions (targetscanCNN/Cons/NonCons) ===")

    summary_counts_path = pathlib.Path(summary_counts_path)
    if not summary_counts_path.exists():
        raise FileNotFoundError(f"Summary counts file not found: {summary_counts_path}")

    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = utr_index["best_tx_by_gene_id"]
    gene_id_by_tx_ts: Dict[str, str] = utr_index["gene_id_by_tx"]

    tx_to_gene_ens = ensembl_tables["tx_to_gene"]
    gene_to_name_ens = ensembl_tables["gene_to_name"]

    out_predictions_dir = pathlib.Path(out_predictions_dir)
    out_predictions_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "Transcript ID",
        "Gene Symbol",
        "miRNA family",
        "Species ID",
        "Representative miRNA",
        "Total num conserved sites",
        "Total num nonconserved sites",
        score_col,
    ]

    # Pass 1: compute min/max on final CNN set (after all mandatory filters)
    minv = math.inf
    maxv = -math.inf

    stats_minmax = Counter()

    with summary_counts_path.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {summary_counts_path}")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {summary_counts_path}")

        for row in reader:
            stats_minmax["rows_total"] += 1

            if row["Species ID"].strip() != species_id:
                continue
            stats_minmax["rows_species"] += 1

            tx_raw = row["Transcript ID"].strip()
            gene_id_ts = gene_id_by_tx_ts.get(tx_raw)
            if gene_id_ts is None:
                stats_minmax["drop_tx_not_in_utr_sequences"] += 1
                continue
            stats_minmax["rows_tx_mapped_ts"] += 1

            best = best_tx_by_gene_id.get(gene_id_ts)
            if best is None or tx_raw != best[0]:
                stats_minmax["drop_not_longest_utr"] += 1
                continue
            stats_minmax["rows_longest_utr"] += 1

            tx_stable = _strip_version(tx_raw)
            gene_id_ens = tx_to_gene_ens.get(tx_stable)
            if gene_id_ens is None:
                stats_minmax["drop_tx_not_in_ensembl"] += 1
                continue
            stats_minmax["rows_tx_in_ensembl"] += 1

            fam = row["miRNA family"].strip()
            mirs = family_to_mirs.get(fam)
            if not mirs:
                stats_minmax["drop_family_no_human_mirs"] += 1
                continue
            stats_minmax["rows_family_mapped"] += 1

            v = _to_float_or_none(row[score_col])
            if v is None:
                stats_minmax["drop_null_score"] += 1
                continue
            stats_minmax["rows_scored"] += 1

            if v < minv:
                minv = v
            if v > maxv:
                maxv = v

    if stats_minmax["rows_scored"] == 0:
        raise ValueError(
            "No scored rows remain after mandatory filters. "
            "Check transcript mapping, family mapping, and score column."
        )

    logger.info(
        "Normalization params for %s (FINAL filtered CNN rows): min=%.6g max=%.6g (scored_rows=%d)",
        score_col, float(minv), float(maxv), stats_minmax["rows_scored"]
    )
    logger.info(
        "Min/max filter diagnostics: total=%d | species=%d | longest_utr=%d | in_ensembl=%d | family_mapped=%d | NULL_score=%d | scored=%d",
        stats_minmax["rows_total"],
        stats_minmax["rows_species"],
        stats_minmax["rows_longest_utr"],
        stats_minmax["rows_tx_in_ensembl"],
        stats_minmax["rows_family_mapped"],
        stats_minmax["drop_null_score"],
        stats_minmax["rows_scored"],
    )

    denom = (maxv - minv)
    if denom <= 0:
        raise ValueError(f"Invalid normalization range: min={minv}, max={maxv}")

    # Prepare writers for 3 predictors
    predictors = ["targetscanCNN", "targetscanCons", "targetscanNonCons"]
    writers: Dict[str, Tuple[csv.DictWriter, Any]] = {}

    def _open_writer(pred: str) -> Tuple[csv.DictWriter, Any, pathlib.Path]:
        out_dir = out_predictions_dir / pred
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pred}_standardized.tsv"
        fout = out_path.open("w", encoding="utf-8", newline="")
        w = csv.DictWriter(
            fout,
            fieldnames=["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"],
            delimiter="\t",
        )
        w.writeheader()
        return w, fout, out_path

    out_paths: Dict[str, pathlib.Path] = {}
    for pred in predictors:
        w, fout, p = _open_writer(pred)
        writers[pred] = (w, fout)
        out_paths[pred] = p

    # Pass 2: write rows + log drops per split
    split_stats: Dict[str, Counter] = {pred: Counter() for pred in predictors}
    global_stats = Counter()

    with summary_counts_path.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        assert reader.fieldnames is not None

        for row in reader:
            global_stats["rows_total"] += 1
            if row["Species ID"].strip() != species_id:
                continue
            global_stats["rows_species"] += 1

            tx_raw = row["Transcript ID"].strip()
            gene_id_ts = gene_id_by_tx_ts.get(tx_raw)
            if gene_id_ts is None:
                global_stats["drop_tx_not_in_utr_sequences"] += 1
                continue
            global_stats["rows_tx_mapped_ts"] += 1

            best = best_tx_by_gene_id.get(gene_id_ts)
            if best is None or tx_raw != best[0]:
                global_stats["drop_not_longest_utr"] += 1
                continue
            global_stats["rows_longest_utr"] += 1

            tx_stable = _strip_version(tx_raw)
            gene_id_ens = tx_to_gene_ens.get(tx_stable)
            if gene_id_ens is None:
                global_stats["drop_tx_not_in_ensembl"] += 1
                continue
            global_stats["rows_tx_in_ensembl"] += 1

            gene_name = gene_to_name_ens.get(gene_id_ens) or row["Gene Symbol"].strip()

            fam = row["miRNA family"].strip()
            mirs = family_to_mirs.get(fam)
            if not mirs:
                global_stats["drop_family_no_human_mirs"] += 1
                continue
            global_stats["rows_family_mapped"] += 1

            v = _to_float_or_none(row[score_col])
            if v is None:
                global_stats["drop_null_score"] += 1
                continue
            global_stats["rows_scored"] += 1

            score_norm = (v - minv) / denom
            if score_norm < 0.0:
                score_norm = 0.0
            elif score_norm > 1.0:
                score_norm = 1.0

            # split membership
            try:
                n_cons = int(float(row["Total num conserved sites"]))
            except Exception:
                n_cons = 0
            try:
                n_noncons = int(float(row["Total num nonconserved sites"]))
            except Exception:
                n_noncons = 0

            in_cnn = True
            in_cons = n_cons > 0
            in_noncons = n_noncons > 0

            # expand family -> all human mature miRNAs
            for mir in mirs:
                if in_cnn:
                    split_stats["targetscanCNN"]["written_rows"] += 1
                    writers["targetscanCNN"][0].writerow(
                        {
                            "Ensembl_ID": gene_id_ens,
                            "Gene_Name": gene_name,
                            "miRNA_ID": mir.mirna_id,
                            "miRNA_Name": mir.mirna_name,
                            "Score": f"{score_norm:.6g}",
                        }
                    )
                if in_cons:
                    split_stats["targetscanCons"]["written_rows"] += 1
                    writers["targetscanCons"][0].writerow(
                        {
                            "Ensembl_ID": gene_id_ens,
                            "Gene_Name": gene_name,
                            "miRNA_ID": mir.mirna_id,
                            "miRNA_Name": mir.mirna_name,
                            "Score": f"{score_norm:.6g}",
                        }
                    )
                if in_noncons:
                    split_stats["targetscanNonCons"]["written_rows"] += 1
                    writers["targetscanNonCons"][0].writerow(
                        {
                            "Ensembl_ID": gene_id_ens,
                            "Gene_Name": gene_name,
                            "miRNA_ID": mir.mirna_id,
                            "miRNA_Name": mir.mirna_name,
                            "Score": f"{score_norm:.6g}",
                        }
                    )

            # track how many base rows are eligible for each split (pre-expansion)
            split_stats["targetscanCNN"]["base_rows"] += 1
            if in_cons:
                split_stats["targetscanCons"]["base_rows"] += 1
            if in_noncons:
                split_stats["targetscanNonCons"]["base_rows"] += 1

    # Close writers
    for pred in predictors:
        writers[pred][1].close()

    # Log final stats
    logger.info("Global filter stats (applied to all splits):")
    logger.info(
        "  total=%d | species=%d | ts_tx_mapped=%d | longest_utr=%d | in_ensembl=%d | family_mapped=%d | NULL_score=%d | scored=%d",
        global_stats["rows_total"],
        global_stats["rows_species"],
        global_stats["rows_tx_mapped_ts"],
        global_stats["rows_longest_utr"],
        global_stats["rows_tx_in_ensembl"],
        global_stats["rows_family_mapped"],
        global_stats["drop_null_score"],
        global_stats["rows_scored"],
    )
    logger.info("  dropped: tx_not_in_utr=%d | not_longest_utr=%d | tx_not_in_ensembl=%d | family_no_human_mirs=%d | NULL_score=%d",
                global_stats["drop_tx_not_in_utr_sequences"],
                global_stats["drop_not_longest_utr"],
                global_stats["drop_tx_not_in_ensembl"],
                global_stats["drop_family_no_human_mirs"],
                global_stats["drop_null_score"])

    for pred in predictors:
        logger.info(
            "%s -> wrote %d rows (family-expanded) from %d base rows. output=%s",
            pred,
            split_stats[pred]["written_rows"],
            split_stats[pred]["base_rows"],
            out_paths[pred],
        )


# =============================================================================
# main workflow
# =============================================================================

# =============================================================================
# EXTRA — Download miRBase latest mature.fa and map IDs -> latest names
# =============================================================================
def download_mirbase_mature(data_dir: pathlib.Path, *, force: bool = False) -> pathlib.Path:
    logger.info("\n=== EXTRA: Download miRBase mature.fa (latest) ===")
    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://mirbase.org/ftp/22.2/mature.fa.gz"
    dest = data_dir / "mirbase_mature.fa"

    if dest.exists() and not force:
        logger.info("Skipping mature.fa (already exists)")
        return dest

    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(dest)
        logger.info("Downloaded miRBase mature.fa")
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

    return dest


def parse_mirbase_mature(mature_fa: pathlib.Path) -> Dict[str, str]:
    """
    Parse mature.fa header lines:
      >MIMAT0000062 hsa-miR-21-5p ...
    Returns: accession -> latest miRNA name
    """
    logger.info("Parsing miRBase mature.fa")
    acc2name = {}
    with mature_fa.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    acc = parts[0]
                    name = parts[1]
                    acc2name[acc] = name
    logger.info("Parsed %d mature miRNAs from miRBase", len(acc2name))
    return acc2name


# =============================================================================
# EXTRA — Final statistics reporting
# =============================================================================
def compute_final_statistics(predictions_root: pathlib.Path):
    logger.info("\n=== FINAL STATISTICS ===")

    sets = ["targetscanCNN", "targetscanCons", "targetscanNonCons"]
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

        logger.info("%s: %d unique genes | %d unique miRNAs",
                    s, len(genes), len(mirs))

    common_genes = set.intersection(*genes_by_set.values())
    common_mirs = set.intersection(*mirs_by_set.values())

    logger.info("Overlap (all three sets): %d genes | %d miRNAs",
                len(common_genes), len(common_mirs))


def main() -> None:
    repo_root = _repo_root()

    # predictor-local data cache
    targetscan_dir = repo_root / "pipelines" / "standardized_predictors" / "targetscan"
    data_dir = targetscan_dir / "data"

    # repo-global predictions output (not tracked)
    out_predictions_dir = repo_root / "data" / "predictions"

    files = step1_download_targetscan_files(data_dir, force=False)

    utr_index = step2_build_longest_utr_index(
        files["UTR_Sequences.txt"],
        species_id="9606",
    )

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

    mirbase_fa = download_mirbase_mature(data_dir, force=False)
    mirbase_latest = parse_mirbase_mature(mirbase_fa)

    family_to_mirs = step_mirfamily_to_human_matures(
        files["miR_Family_Info.txt"],
        species_id="9606",
    )

    step6_write_standardized_predictions(
        files["Summary_Counts.all_predictions.txt"],
        utr_index=utr_index,
        ensembl_tables=ensembl_tables,
        family_to_mirs=family_to_mirs,
        out_predictions_dir=out_predictions_dir,
        species_id="9606",
        score_col="Predicted occupancy - transfected miRNA",
    )

    compute_final_statistics(out_predictions_dir)


if __name__ == "__main__":
    main()
