import logging
import gzip
import csv
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from funmirbench.logger import (
    DEFAULT_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    parse_log_level,
    setup_logging,
)

logger = logging.getLogger("utils")

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def resolve_path_relative_to_root(path: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root())
    except ValueError:
        return path

def configure_logging(log_path: Path, log_level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(parse_log_level(log_level))
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
    )
    root_logger.addHandler(file_handler)

def _log_row_count_change(label: str, before: int, after: int) -> None:
    logger.info("%s: %d -> %d rows", label, before, after)

def _zenodo_record_id(url: str) -> str | None:
    """Record id from a DOI, ``zenodo:ID`` string, or any zenodo.org URL."""
    for prefix in ("10.5281/zenodo.", "zenodo:"):
        if url.startswith(prefix):
            tail = url[len(prefix):]
            return tail if tail.isdigit() else None
    if "zenodo.org" in url:
        for part in reversed(url.rstrip("/").split("/")):
            if part.isdigit():
                return part
    return None


def _file_name(f: dict) -> str | None:
    return f.get("key") or f.get("filename")

def download_file(
    url: str,
    output_path: Path,
    params: Optional[dict[str, str]] = None,
    timeout: int = 120,
    resource_label: str = "file",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = resolve_path_relative_to_root(output_path)

    if output_path.exists():
        logger.info("Using cached %s: %s", resource_label, relative_path)
        return output_path

    logger.info("Downloading %s: %s", resource_label, relative_path)

    record_id = _zenodo_record_id(url)
    if record_id is not None:
        api_url = f"https://zenodo.org/api/records/{record_id}"
        try:
            resp = requests.get(api_url, timeout=timeout)
            resp.raise_for_status()
            rec_json = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch Zenodo record {record_id}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from Zenodo API for record {record_id}") from exc

        files = rec_json.get("files") or []
        if not files:
            raise RuntimeError(f"Zenodo record {record_id} contains no files")

        target = output_path.name
        chosen = next((f for f in files if _file_name(f) == target), None)
        if chosen is None:
            if len(files) == 1:
                chosen = files[0]
            else:
                available = [_file_name(f) for f in files]
                raise RuntimeError(
                    f"Zenodo record {record_id} contains multiple files: {available}. "
                    f"Expected cached filename: {target}."
                )

        filename = _file_name(chosen)
        links = chosen.get("links") or {}
        download_url = (
            links.get("download")
            or links.get("self")
            or f"https://zenodo.org/record/{record_id}/files/{filename}?download=1"
        )

        try:
            with requests.get(download_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(output_path, "wb") as out_f:
                    for chunk in r.iter_content(chunk_size=8192):
                        out_f.write(chunk)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download {resource_label} from Zenodo record {record_id}: {exc}"
            ) from exc

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Empty response downloading {resource_label} from Zenodo record {record_id}"
            )

        logger.info("Saved %s: %s", resource_label, relative_path)
        return output_path

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download {resource_label} from {url} "
            f"to {relative_path}: {exc}"
        ) from exc
    if not response.content.strip():
        raise RuntimeError(
            f"Failed to download {resource_label} from {url} "
            f"to {relative_path}: empty response"
        )
    response_text = response.content.decode("utf-8", errors="replace")
    if "Query ERROR:" in response_text or "BioMart::Exception" in response_text:
        raise RuntimeError(
            f"Failed to download {resource_label} from {url} "
            f"to {relative_path}: BioMart returned an error response"
        )

    output_path.write_bytes(response.content)
    logger.info("Saved %s: %s", resource_label, relative_path)
    return output_path

def load_prediction_files(
    path: Path,
    raw_gene_ID_column: str,
    raw_gene_Name_column: str,
    raw_mirna_column: str,
    raw_prediction_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        usecols=[0, 1, 2, 3],
        names=[
            raw_gene_ID_column,
            raw_gene_Name_column,
            raw_mirna_column,
            raw_prediction_column,
        ],
        dtype=str,
    )
    if df.empty:
        raise RuntimeError("No prediction file was loaded")

    return df

def load_ensembl_tx_to_gene(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Ensembl transcript-to-gene file: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    mapping: dict[str, str] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        table = pd.read_csv(handle, sep="\t", dtype=str)

    missing = [col for col in ("transcript_id", "gene_id") if col not in table.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    for _, row in table.iterrows():
        tx_id = str(row["transcript_id"]).split(".", 1)[0]
        gene_id = str(row["gene_id"]).split(".", 1)[0]
        if not tx_id or tx_id == "nan" or not gene_id or gene_id == "nan":
            continue
        if tx_id in mapping and mapping[tx_id] != gene_id:
            raise ValueError(
                f"{path}: conflicting gene mappings for transcript {tx_id}: "
                f"{mapping[tx_id]} vs {gene_id}"
            )
        mapping[tx_id] = gene_id
    return mapping

def build_ensembl_tx_to_gene_from_gtf(gtf_gz_path: Path, tx2gene_path: Path) -> dict[str, str]:
    if not gtf_gz_path.exists():
        raise FileNotFoundError(f"Missing Ensembl GTF file: {gtf_gz_path}")

    tx2gene_path.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    n_transcript_rows = 0
    with gzip.open(gtf_gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            n_transcript_rows += 1

            attrs = {}
            for part in [item.strip() for item in fields[8].strip().split(";") if item.strip()]:
                if " " not in part:
                    continue
                key, value = part.split(" ", 1)
                attrs[key] = value.strip().strip('"')

            transcript_id = attrs.get("transcript_id")
            gene_id = attrs.get("gene_id")
            if not transcript_id or not gene_id:
                continue
            mapping[transcript_id.split(".", 1)[0]] = gene_id.split(".", 1)[0]

    with gzip.open(tx2gene_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transcript_id", "gene_id"], delimiter="\t")
        writer.writeheader()
        for transcript_id, gene_id in sorted(mapping.items()):
            writer.writerow({"transcript_id": transcript_id, "gene_id": gene_id})

    logger.info(
        "Built Ensembl transcript-to-gene cache from GTF: transcript_rows=%d | mapped=%d | output=%s",
        n_transcript_rows,
        len(mapping),
        resolve_path_relative_to_root(tx2gene_path),
    )
    return mapping

def map_transcripts_to_genes(
    df: pd.DataFrame,
    tx_to_gene: dict[str, str],
    transcript_column: str,
    gene_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[transcript_column] = out[transcript_column].astype(str).str.strip()
    transcript_ids = out[transcript_column].str.split(".", n=1).str[0]
    out[gene_column] = transcript_ids.map(tx_to_gene)
    return _drop_unmapped_rows(
        out,
        gene_column,
        "Drop prediction rows with Ensembl transcripts that cannot map to genes",
    )

def collapse_transcript_rows_to_genes(
    df: pd.DataFrame,
    gene_column: str,
    gene_name_column: str,
    mirna_name_column: str,
    mimat_column: str,
    score_column: str,
) -> pd.DataFrame:
    before = len(df)
    duplicate_mask = df.duplicated([gene_column, mirna_name_column], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        duplicate_pairs = (
            df.loc[duplicate_mask]
            .groupby([gene_column, mirna_name_column], dropna=False)[score_column]
            .nunique()
        )
        conflicting_pairs = int((duplicate_pairs > 1).sum())
        logger.info(
            "Transcript-to-gene collapse found %d duplicate rows across %d gene-miRNA "
            "pairs; %d pairs had conflicting scores",
            duplicate_rows,
            len(duplicate_pairs),
            conflicting_pairs,
        )

    sort_columns = [gene_column, mirna_name_column, score_column]
    out = (
        df.sort_values(sort_columns, ascending=[True, True, False])
        .drop_duplicates([gene_column, mirna_name_column], keep="first")
        .loc[:, [gene_column, gene_name_column, mimat_column, mirna_name_column, score_column]]
        .copy()
    )
    _log_row_count_change(
        "Collapse transcript rows to one row per Ensembl gene-miRNA pair",
        before,
        len(out),
    )
    return out

def create_mirna_name_to_mimat_mapping(mature_fa_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}

    with open(mature_fa_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                continue

            parts = line[1:].strip().split()
            if len(parts) < 2:
                raise ValueError(f"{mature_fa_path}: invalid FASTA header; {line.strip()}")

            mirna_name, mimat_id = parts[0], parts[1]

            if not mirna_name.startswith("hsa-"):
                continue

            if not mimat_id.startswith("MIMAT"):
                raise ValueError(f"{mature_fa_path}: invalid MIMAT ID; {line.strip()}")

            if mirna_name in mapping and mapping[mirna_name] != mimat_id:
                raise ValueError(
                    f"{mature_fa_path}: conflicting MIMAT for {mirna_name}: "
                    f"{mapping[mirna_name]} vs {mimat_id}"
                )

            mapping[mirna_name] = mimat_id

    return mapping

def _drop_unmapped_rows(
    df: pd.DataFrame,
    mapped_column: str,
    log_label: str,
) -> pd.DataFrame:
    before = len(df)
    out = df.loc[df[mapped_column].notna()].copy()
    _log_row_count_change(log_label, before, len(out))
    return out

def map_mirna_names_to_mimat(
    df: pd.DataFrame,
    mirna_name_to_id: dict[str, str],
    query_column: str,
    mirna_name_column: str,
    mimat_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[mirna_name_column] = out[query_column].astype(str).str.strip()
    out[mimat_column] = out[mirna_name_column].map(mirna_name_to_id)
    return _drop_unmapped_rows(
        out,
        mimat_column,
        "Drop prediction rows with miRNA names that cannot map to MIMAT IDs",
    )

def build_output_table(
    df: pd.DataFrame,
    prediction_column: str,
    score_column: str,
    final_columns: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out[score_column] = pd.to_numeric(out[prediction_column], errors="coerce")
    before = len(out)
    out = out.loc[out[score_column].notna()].copy()
    _log_row_count_change("Drop prediction rows with non-numeric scores", before, len(out))
    return out.loc[:, final_columns].copy()
