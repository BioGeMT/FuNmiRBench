import pathlib
import urllib.request
import shutil
import zipfile
import logging
import csv
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_float_or_none(x: str) -> Optional[float]:
    x = x.strip()
    if x == "" or x.upper() == "NULL":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _utr_len(seq: str) -> int:
    """Count UTR length ignoring gaps and non-ACGTU chars."""
    s = seq.strip().upper()
    return sum(1 for ch in s if ch in ("A", "C", "G", "U", "T"))


# =============================================================================
# STEP 1/4 — Download + unzip (flat under data/)
# =============================================================================
def step_download_targetscan_files(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> Dict[str, pathlib.Path]:
    """
    Download the minimal set of TargetScan v8.0 files needed for:
      - longest-UTR transcript selection (UTR_Sequences)
      - gene-level predictions + scores (Summary_Counts.all_predictions)
      - later family→miRNA mapping (miR_Family_Info)

    Files are stored FLAT under:
      pipelines/standarized_predictors/targetscan/data/
    """
    logger.info("\n=== STEP 1/4: Download + unzip TargetScan (flat under data/) ===")

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
            if tmp.exists():
                tmp.unlink(missing_ok=True)

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

    logger.info("✔ TargetScan files ready in %s", data_dir)
    return expected


# =============================================================================
# STEP 2/4 — Build longest-UTR transcript index (TargetScan-consistent)
# =============================================================================
def step_build_longest_utr_index(
    utr_sequences_path: pathlib.Path,
    *,
    species_id: str = "9606",
    report_top_n_tx_counts: int = 6,
) -> Dict[str, Any]:
    """
    TargetScan predictions are transcript-level. For gene-level benchmarking we pick
    ONE transcript per gene: the transcript with the LONGEST annotated UTR.

    We compute UTR lengths from TargetScan's own UTR_Sequences file to avoid
    mismatches between our transcript annotations and the transcript universe used
    by TargetScan.
    """
    logger.info("\n=== STEP 2/4: Build longest-UTR transcript index from UTR_Sequences.txt ===")

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
        "Longest-UTR ties: genes_with_tie=%d (%.2f%% of genes_with_UTR). Tie rule: keep first-seen transcript among tied max length.",
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
# STEP 3/4 — Compute min/max for score normalization (after filters)
# =============================================================================
def step_compute_minmax_for_score(
    summary_counts_path: pathlib.Path,
    utr_index: Dict[str, Any],
    *,
    species_id: str = "9606",
    score_col: str = "Predicted occupancy - transfected miRNA",
) -> Tuple[float, float, Dict[str, int]]:
    logger.info("\n=== STEP 3/4: Compute min/max for normalization (after species+longest-UTR filters) ===")

    summary_counts_path = pathlib.Path(summary_counts_path)
    if not summary_counts_path.exists():
        raise FileNotFoundError(f"Summary counts file not found: {summary_counts_path}")

    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = utr_index["best_tx_by_gene_id"]
    gene_id_by_tx: Dict[str, str] = utr_index["gene_id_by_tx"]

    minv = math.inf
    maxv = -math.inf

    stats = {
        "total_rows": 0,
        "species_rows": 0,
        "tx_mapped": 0,
        "longest_tx_rows": 0,
        "non_null_scores": 0,
        "null_scores": 0,
    }

    with summary_counts_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {summary_counts_path}")

        required = ["Transcript ID", "Species ID", score_col]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {summary_counts_path}")

        for row in reader:
            stats["total_rows"] += 1
            if row["Species ID"].strip() != species_id:
                continue
            stats["species_rows"] += 1

            tx = row["Transcript ID"].strip()
            gene_id = gene_id_by_tx.get(tx)
            if gene_id is None:
                continue
            stats["tx_mapped"] += 1

            best = best_tx_by_gene_id.get(gene_id)
            if best is None or tx != best[0]:
                continue
            stats["longest_tx_rows"] += 1

            v = _to_float_or_none(row[score_col])
            if v is None:
                stats["null_scores"] += 1
                continue
            stats["non_null_scores"] += 1
            minv = min(minv, v)
            maxv = max(maxv, v)

    if stats["non_null_scores"] == 0:
        raise ValueError(
            f"No non-NULL scores found for {score_col!r} after filtering. "
            "Check that transcript IDs match between Summary_Counts and UTR_Sequences."
        )

    logger.info(
        "Normalization params for %s: min=%.6g max=%.6g (non_NULL_scores=%d, NULL_scores=%d)",
        score_col, float(minv), float(maxv), stats["non_null_scores"], stats["null_scores"]
    )
    logger.info(
        "Filter stats: total_rows=%d | species_rows=%d | tx_mapped=%d | longest_tx_rows=%d",
        stats["total_rows"], stats["species_rows"], stats["tx_mapped"], stats["longest_tx_rows"]
    )
    return float(minv), float(maxv), stats


# =============================================================================
# STEP 4/4 — Write one intermediate base table (human + longest UTR)
# =============================================================================
def step_write_base_table_human_longest_utr(
    summary_counts_path: pathlib.Path,
    utr_index: Dict[str, Any],
    *,
    out_path: pathlib.Path,
    species_id: str = "9606",
    score_col: str = "Predicted occupancy - transfected miRNA",
    min_score: float,
    max_score: float,
) -> Dict[str, int]:
    logger.info("\n=== STEP 4/4: Write base table (human + longest UTR + normalized score) ===")

    summary_counts_path = pathlib.Path(summary_counts_path)
    out_path = pathlib.Path(out_path)

    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = utr_index["best_tx_by_gene_id"]
    gene_id_by_tx: Dict[str, str] = utr_index["gene_id_by_tx"]

    denom = (max_score - min_score)
    if denom <= 0:
        raise ValueError(f"Invalid normalization range: min={min_score}, max={max_score}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_rows": 0,
        "species_rows": 0,
        "tx_mapped": 0,
        "longest_tx_rows": 0,
        "null_scores": 0,
        "written": 0,
    }

    with summary_counts_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {summary_counts_path}")

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
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {summary_counts_path}")

        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "Transcript_ID",
                "Gene_ID",
                "Gene_Symbol",
                "miRNA_family",
                "Representative_miRNA",
                "Total_num_conserved_sites",
                "Total_num_nonconserved_sites",
                "Score_raw",
                "Score_norm",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        for row in reader:
            stats["total_rows"] += 1
            if row["Species ID"].strip() != species_id:
                continue
            stats["species_rows"] += 1

            tx = row["Transcript ID"].strip()
            gene_id = gene_id_by_tx.get(tx)
            if gene_id is None:
                continue
            stats["tx_mapped"] += 1

            best = best_tx_by_gene_id.get(gene_id)
            if best is None or tx != best[0]:
                continue
            stats["longest_tx_rows"] += 1

            v = _to_float_or_none(row[score_col])
            if v is None:
                stats["null_scores"] += 1
                continue

            norm = (v - min_score) / denom
            norm = 0.0 if norm < 0.0 else (1.0 if norm > 1.0 else norm)

            writer.writerow(
                {
                    "Transcript_ID": tx,
                    "Gene_ID": gene_id,
                    "Gene_Symbol": row["Gene Symbol"].strip(),
                    "miRNA_family": row["miRNA family"].strip(),
                    "Representative_miRNA": row["Representative miRNA"].strip(),
                    "Total_num_conserved_sites": row["Total num conserved sites"].strip(),
                    "Total_num_nonconserved_sites": row["Total num nonconserved sites"].strip(),
                    "Score_raw": f"{v:.6g}",
                    "Score_norm": f"{norm:.6g}",
                }
            )
            stats["written"] += 1

    logger.info("Wrote base table: %s", out_path)
    logger.info(
        "Write stats: total_rows=%d | species_rows=%d | tx_mapped=%d | longest_tx_rows=%d | NULL_scores=%d | written=%d",
        stats["total_rows"], stats["species_rows"], stats["tx_mapped"],
        stats["longest_tx_rows"], stats["null_scores"], stats["written"]
    )
    return stats


def main() -> None:
    targetscan_dir = pathlib.Path("pipelines/standarized_predictors/targetscan")
    data_dir = targetscan_dir / "data"
    out_base = data_dir / "targetscan_human_longestutr_base.tsv"

    files = step_download_targetscan_files(data_dir, force=False)

    utr_index = step_build_longest_utr_index(
        files["UTR_Sequences.txt"],
        species_id="9606",
    )

    score_col = "Predicted occupancy - transfected miRNA"
    min_score, max_score, _ = step_compute_minmax_for_score(
        files["Summary_Counts.all_predictions.txt"],
        utr_index,
        species_id="9606",
        score_col=score_col,
    )

    step_write_base_table_human_longest_utr(
        files["Summary_Counts.all_predictions.txt"],
        utr_index,
        out_path=out_base,
        species_id="9606",
        score_col=score_col,
        min_score=min_score,
        max_score=max_score,
    )


if __name__ == "__main__":
    main()
