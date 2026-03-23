#!/usr/bin/env python3
"""
TargetScan v8 (vert_80) standardization pipeline (human).

What this script does
---------------------
1) Downloads the TargetScan input files needed to reproduce the pipeline:
   - Summary_Counts.all_predictions.txt
   - miR_Family_Info.txt
   - Gene_info.txt

2) Builds a representative-transcript-per-gene index from TargetScan Gene_info for species=9606.
   The representative transcript is taken directly from TargetScan's annotation
   (Representative transcript? = 1).

3) Downloads Ensembl v115 GTF (GRCh38) and builds cached mapping tables:
   - transcript_id (ENST stable) -> gene_id (ENSG stable)
   - gene_id (ENSG stable) -> gene_name

4) Runs QC overlap diagnostics against Ensembl v115 transcript IDs (after stripping version suffixes).

5) Downloads miRBase mature.fa pinned to release 22.1 and parses accession->name mapping.

6) Builds one standardized prediction set from Summary_Counts.all_predictions.txt:
   - targetscan (score: Cumulative weighted context++ score)

Score handling
--------------
- Score is kept RAW from TargetScan.
- Score_norm is direction-standardized and rank-normalized so that higher means stronger.
- For TargetScan, stronger means more negative raw score, so ranking is computed on -1 * raw score.
- Ranking is computed at the base-row level before family expansion, so miRNA family size does not affect ranks.

Outputs
-------
Standardized TSV is written to:
  data/predictions/targetscan/targetscan_standardized.tsv

Schema:
  Ensembl_ID, Gene_Name, miRNA_ID, miRNA_Name, Score, Score_norm
"""

from __future__ import annotations

import csv
import gzip as gz
import logging
import pathlib
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


logger = logging.getLogger(__name__)


MIRBASE_RELEASE = "22.1"
MIRBASE_MATURE_URL = "https://mirbase.org/download_version_files/22.1/mature.fa"
PREDICTOR_NAME = "targetscan"
TARGETSCAN_SCORE_COL = "Cumulative weighted context++ score"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _strip_version(x: str) -> str:
    s = x.strip()
    return s.split(".", 1)[0] if "." in s else s


def _to_float_or_none(x: str):
    x = (x or "").strip()
    if x == "" or x.upper() == "NULL":
        return None
    return float(x)


def _download_url(url: str, dest: pathlib.Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as out:
            shutil.copyfileobj(r, out)
        tmp.replace(dest)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _assert_fasta(path: pathlib.Path) -> None:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith(">"):
                raise RuntimeError(f"Not FASTA (first line: {s[:120]})")
            return
    raise RuntimeError("FASTA check failed: file appears empty.")


def _percentile_ranks(values: List[float]) -> List[float]:
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
            ranks[indexed[k][0]] = pct

        i = j

    return ranks


def setup_logging(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def step1_download_targetscan_files(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> Dict[str, pathlib.Path]:
    logger.info("\n=== STEP 1/7: Download + unzip TargetScan inputs ===")

    data_dir = pathlib.Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    base = "https://www.targetscan.org/vert_80/vert_80_data_download"
    zips = [
        "Summary_Counts.all_predictions.txt.zip",
        "miR_Family_Info.txt.zip",
        "Gene_info.txt.zip",
    ]

    for fname in zips:
        url = f"{base}/{fname}"
        zip_path = data_dir / fname

        if not zip_path.exists() or force:
            logger.info("Downloading %s", url)
            _download_url(url, zip_path)
            logger.info("Downloaded %s", zip_path.name)
        else:
            logger.info("Skipping %s (already exists)", zip_path.name)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                extracted = data_dir / member
                if extracted.exists() and not force:
                    continue
                zf.extract(member, data_dir)
        logger.info("Unzipped %s", zip_path.name)

    expected = {
        "Summary_Counts.all_predictions.txt": data_dir / "Summary_Counts.all_predictions.txt",
        "miR_Family_Info.txt": data_dir / "miR_Family_Info.txt",
        "Gene_info.txt": data_dir / "Gene_info.txt",
    }

    logger.info("✔ TargetScan inputs ready in %s", data_dir)
    return expected


def step2_build_representative_transcript_index(
    gene_info_path: pathlib.Path,
    *,
    species_id: str = "9606",
    report_top_n: int = 6,
) -> Dict[str, Any]:
    logger.info("\n=== STEP 2/7: Build representative-transcript index from Gene_info.txt ===")

    best_tx_by_gene_id: Dict[str, Tuple[str, int, str]] = {}
    gene_id_by_tx: Dict[str, str] = {}
    gene_symbol_by_gene_id: Dict[str, str] = {}

    tag_count_dist: Counter[int] = Counter()
    tx_count_by_gene: Counter[str] = Counter()
    rep_count_by_gene: Counter[str] = Counter()

    n_total = 0
    n_species = 0
    n_rep_rows = 0

    with pathlib.Path(gene_info_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            n_total += 1
            if row["Species ID"].strip() != species_id:
                continue
            n_species += 1

            tx = row["Transcript ID"].strip()
            gene_id = row["Gene ID"].strip()
            gene_sym = row["Gene symbol"].strip()
            rep_flag = row["Representative transcript?"].strip()
            tags = int(row["3P-seq tags"].strip())

            gene_id_by_tx[tx] = gene_id
            gene_symbol_by_gene_id[gene_id] = gene_sym
            tx_count_by_gene[gene_id] += 1

            if rep_flag != "1":
                continue

            n_rep_rows += 1
            rep_count_by_gene[gene_id] += 1
            tag_count_dist[tags] += 1

            if gene_id not in best_tx_by_gene_id:
                best_tx_by_gene_id[gene_id] = (tx, tags, gene_sym)

    genes_with_rep = len(best_tx_by_gene_id)
    genes_with_multi_rep = sum(1 for c in rep_count_by_gene.values() if c > 1)

    logger.info(
        "Gene_Info stats: rows_total=%d | rows_species(%s)=%d | representative_rows=%d | genes_with_representative_tx=%d",
        n_total, species_id, n_species, n_rep_rows, genes_with_rep
    )

    top_bins = Counter(tx_count_by_gene.values()).most_common(report_top_n)
    logger.info(
        "Gene_Info transcripts-per-gene (top %d bins): %s",
        report_top_n,
        " | ".join([f"{k}tx:{v}" for k, v in top_bins]),
    )

    logger.info(
        "Representative-transcript QC: genes_with_multiple_representative_rows=%d",
        genes_with_multi_rep
    )

    top_tags = tag_count_dist.most_common(report_top_n)
    logger.info(
        "Representative transcript 3P-seq tag counts (top %d bins): %s",
        report_top_n,
        " | ".join([f"{k}:{v}" for k, v in top_tags]),
    )

    for gid, (tx, tags, sym) in list(best_tx_by_gene_id.items())[:3]:
        logger.info("  sample representative transcript: %s -> %s (3P-seq tags=%d, symbol=%s)", gid, tx, tags, sym)

    return {
        "best_tx_by_gene_id": best_tx_by_gene_id,
        "gene_id_by_tx": gene_id_by_tx,
        "gene_symbol_by_gene_id": gene_symbol_by_gene_id,
    }


def step3_download_ensembl115_gtf(
    data_dir: pathlib.Path,
    *,
    force: bool = False,
) -> pathlib.Path:
    logger.info("\n=== STEP 3/7: Download Ensembl v115 GTF (GRCh38) ===")

    data_dir = pathlib.Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz"
    dest = data_dir / "Homo_sapiens.GRCh38.115.gtf.gz"

    if dest.exists() and not force:
        logger.info("Skipping %s (already exists)", dest.name)
        return dest

    logger.info("Downloading %s", url)
    _download_url(url, dest)
    logger.info("Downloaded %s", dest.name)
    return dest


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

    if tx2gene_path.exists() and gene2name_path.exists() and not force_rebuild:
        tx_to_gene: Dict[str, str] = {}
        gene_to_name: Dict[str, str] = {}

        with gz.open(tx2gene_path, "rt", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                tx_to_gene[row["transcript_id"]] = row["gene_id"]

        with gz.open(gene2name_path, "rt", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                gene_to_name[row["gene_id"]] = row["gene_name"]

        logger.info(
            "Loaded cached Ensembl tables: tx_to_gene=%d | gene_to_name=%d",
            len(tx_to_gene), len(gene_to_name)
        )
        return {"tx_to_gene": tx_to_gene, "gene_to_name": gene_to_name}

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
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            n_transcript_rows += 1

            attrs = {}
            for part in [p.strip() for p in fields[8].strip().split(";") if p.strip()]:
                if " " in part:
                    k, v = part.split(" ", 1)
                    attrs[k] = v.strip().strip('"')

            gene_id = attrs.get("gene_id")
            tx_id = attrs.get("transcript_id")
            gene_name = attrs.get("gene_name")

            if gene_id and tx_id:
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


def step5_qc_targetscan_vs_ensembl_transcripts(
    tx_index: Dict[str, Any],
    ensembl_tables: Dict[str, Dict[str, str]],
    *,
    report_n: int = 10,
) -> None:
    logger.info("\n=== STEP 5/7: QC TargetScan representative transcript overlap with Ensembl v115 ===")

    gene_id_by_tx = tx_index["gene_id_by_tx"]
    best_tx_by_gene_id = tx_index["best_tx_by_gene_id"]
    tx_to_gene_ensembl = ensembl_tables["tx_to_gene"]

    ts_all_tx = list(gene_id_by_tx.keys())
    ts_all_tx_stripped = [_strip_version(t) for t in ts_all_tx]
    stripped_hits = sum(1 for t in ts_all_tx_stripped if t in tx_to_gene_ensembl)

    logger.info("TargetScan Gene_Info transcripts total: %d", len(ts_all_tx))
    logger.info("Ensembl v115 transcripts indexed: %d", len(tx_to_gene_ensembl))
    logger.info("Overlap (stable transcript ID): %d", stripped_hits)

    ts_rep_tx = [tx for (tx, _tags, _sym) in best_tx_by_gene_id.values()]
    ts_rep_tx_stripped = [_strip_version(t) for t in ts_rep_tx]
    rep_hits = sum(1 for t in ts_rep_tx_stripped if t in tx_to_gene_ensembl)

    logger.info("TargetScan representative transcripts: %d", len(ts_rep_tx))
    logger.info("Representative-transcript overlap with Ensembl (stable transcript ID): %d", rep_hits)

    misses = []
    for raw, stripped in zip(ts_rep_tx, ts_rep_tx_stripped):
        if stripped not in tx_to_gene_ensembl:
            misses.append((raw, stripped))
            if len(misses) >= report_n:
                break

    if misses:
        logger.info("Sample representative-transcript misses (raw -> stripped):")
        for raw, stripped in misses:
            logger.info("  %s -> %s", raw, stripped)

    non_enst = [tx for tx in ts_rep_tx_stripped if not tx.startswith("ENST")]
    if non_enst:
        logger.info("Representative transcripts not ENST*: %d", len(non_enst))
        for tx in non_enst[: min(report_n, len(non_enst))]:
            logger.info("  non-ENST example: %s", tx)


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
    _download_url(url=MIRBASE_MATURE_URL, dest=dest)
    _assert_fasta(dest)
    logger.info("Downloaded miRBase mature.fa -> %s", dest)
    return dest


def parse_mirbase_mature(mature_fa: pathlib.Path) -> Dict[str, str]:
    logger.info("Parsing miRBase mature.fa")

    acc2name: Dict[str, str] = {}
    with pathlib.Path(mature_fa).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                mir_name = parts[0]
                mir_acc = parts[1]
                acc2name[mir_acc] = mir_name

    logger.info("Parsed %d mature miRNAs from miRBase", len(acc2name))
    return acc2name


@dataclass(frozen=True)
class MirnaEntry:
    mirna_id: str
    mirna_name: str


def step_mirfamily_to_human_matures(
    mir_family_info_path: pathlib.Path,
    *,
    mirbase_acc2name: Dict[str, str],
    species_id: str = "9606",
) -> Dict[str, List[MirnaEntry]]:
    logger.info("\n=== STEP 7/7 Build miRNA family -> human mature miRNAs mapping (validated against miRBase v22.1) ===")

    fam2mirs: Dict[str, List[MirnaEntry]] = defaultdict(list)
    seen: Dict[str, set[Tuple[str, str]]] = defaultdict(set)

    n_total = 0
    n_species = 0
    n_missing_fields = 0
    n_acc_missing_in_mirbase = 0
    n_name_match = 0
    n_name_replaced = 0
    n_kept = 0

    with pathlib.Path(mir_family_info_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

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


def step_write_standardized_predictions(
    summary_counts_path: pathlib.Path,
    *,
    tx_index: Dict[str, Any],
    ensembl_tables: Dict[str, Dict[str, str]],
    family_to_mirs: Dict[str, List[MirnaEntry]],
    out_predictions_dir: pathlib.Path,
    species_id: str = "9606",
) -> None:
    logger.info("\n=== Write standardized predictions (%s) ===", PREDICTOR_NAME)

    best_tx_by_gene_id = tx_index["best_tx_by_gene_id"]
    gene_id_by_tx_ts = tx_index["gene_id_by_tx"]
    tx_to_gene_ens = ensembl_tables["tx_to_gene"]
    gene_to_name_ens = ensembl_tables["gene_to_name"]

    out_predictions_dir = pathlib.Path(out_predictions_dir)
    out_predictions_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    weakest_nonnull_raw = None

    with pathlib.Path(summary_counts_path).open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")

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
            if tx_stable not in tx_to_gene_ens:
                continue

            fam = row["miRNA family"].strip()
            if fam not in family_to_mirs:
                continue

            stats["rows_after_filters"] += 1
            v = _to_float_or_none(row[TARGETSCAN_SCORE_COL])
            if v is None:
                stats["rows_null_score"] += 1
                continue

            stats["rows_nonnull_score"] += 1
            if weakest_nonnull_raw is None or v > weakest_nonnull_raw:
                weakest_nonnull_raw = v

    if stats["rows_after_filters"] == 0:
        raise ValueError(f"No rows remain after filters for {PREDICTOR_NAME}.")
    if weakest_nonnull_raw is None:
        raise ValueError(f"No non-NULL scores remain after filters for {PREDICTOR_NAME}.")

    logger.info(
        "%s score column '%s': weakest raw(non-NULL)=%.6g | NULL rows=%d (imputed as %.6g)",
        PREDICTOR_NAME,
        TARGETSCAN_SCORE_COL,
        weakest_nonnull_raw,
        stats["rows_null_score"],
        weakest_nonnull_raw,
    )

    rows: List[Dict[str, Any]] = []

    with pathlib.Path(summary_counts_path).open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")

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

            raw_score = _to_float_or_none(row[TARGETSCAN_SCORE_COL])
            if raw_score is None:
                raw_score = weakest_nonnull_raw

            rows.append(
                {
                    "Ensembl_ID": gene_id_ens,
                    "Gene_Name": gene_to_name_ens.get(gene_id_ens) or row["Gene Symbol"].strip(),
                    "mirs": mirs,
                    "score_raw": float(raw_score),
                }
            )

    out_dir = out_predictions_dir / PREDICTOR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{PREDICTOR_NAME}_standardized.tsv"

    ranking_values = [-r["score_raw"] for r in rows]
    rank_scores = _percentile_ranks(ranking_values)

    written = Counter()
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score", "Score_norm"],
            delimiter="\t",
        )
        writer.writeheader()

        for row_obj, rank_score in zip(rows, rank_scores):
            for mir in row_obj["mirs"]:
                writer.writerow(
                    {
                        "Ensembl_ID": row_obj["Ensembl_ID"],
                        "Gene_Name": row_obj["Gene_Name"],
                        "miRNA_ID": mir.mirna_id,
                        "miRNA_Name": mir.mirna_name,
                        "Score": f"{row_obj['score_raw']:.6g}",
                        "Score_norm": f"{rank_score:.6g}",
                    }
                )
                written["written_rows"] += 1
            written["base_rows"] += 1

    logger.info(
        "%s rank normalization: base_rows=%d | min_rank=%.6g | max_rank=%.6g",
        PREDICTOR_NAME,
        len(rows),
        min(rank_scores) if rank_scores else float("nan"),
        max(rank_scores) if rank_scores else float("nan"),
    )
    logger.info(
        "%s -> wrote %d rows (family-expanded) from %d base rows. output=%s",
        PREDICTOR_NAME,
        written["written_rows"],
        written["base_rows"],
        out_path,
    )


def compute_final_statistics(predictions_root: pathlib.Path) -> None:
    logger.info("\n=== FINAL STATISTICS ===")

    p = predictions_root / PREDICTOR_NAME / f"{PREDICTOR_NAME}_standardized.tsv"
    genes = set()
    mirs = set()
    n_rows = 0

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            genes.add(row["Ensembl_ID"])
            mirs.add(row["miRNA_ID"])

    logger.info("%s: %d rows | %d unique genes | %d unique miRNAs", PREDICTOR_NAME, n_rows, len(genes), len(mirs))


def main() -> None:
    repo_root = _repo_root()

    log_file = repo_root / "pipelines" / "standardized_predictors" / "targetscan" / "targetscan_pipeline.log"

    global logger
    logger = setup_logging(log_file)
    logger.info("Logging to file: %s", log_file)

    targetscan_dir = repo_root / "pipelines" / "standardized_predictors" / "targetscan"
    data_dir = targetscan_dir / "data"
    out_predictions_dir = repo_root / "data" / "predictions"

    files = step1_download_targetscan_files(data_dir, force=False)

    tx_index = step2_build_representative_transcript_index(files["Gene_info.txt"], species_id="9606")

    ensembl_gtf = step3_download_ensembl115_gtf(data_dir, force=False)
    ensembl_tables = step4_build_and_cache_ensembl115_tables(
        ensembl_gtf,
        cache_dir=data_dir,
        force_rebuild=False,
    )

    step5_qc_targetscan_vs_ensembl_transcripts(
        tx_index=tx_index,
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
        tx_index=tx_index,
        ensembl_tables=ensembl_tables,
        family_to_mirs=family_to_mirs,
        out_predictions_dir=out_predictions_dir,
        species_id="9606",
    )

    compute_final_statistics(out_predictions_dir)


if __name__ == "__main__":
    main()
