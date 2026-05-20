import logging
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
                    "Set `--predictions-file` to the desired filename or pass a direct URL."
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

def _drop_invalid_rows(
    df: pd.DataFrame,
    columns: list[str],
    log_label: str,
) -> pd.DataFrame:
    
    before = len(df)
    subset = df[columns]
    normalized = subset.astype("string").apply(
        lambda col: col.str.strip().str.lower()
    )
    valid_rows = (
        ~subset.isna().any(axis=1) &          # real NaNs
        ~(normalized == "").any(axis=1) &     # empty after strip
        ~(normalized == "nan").any(axis=1)    # string "nan"
    )
    out = df.loc[valid_rows].copy()

    _log_row_count_change(log_label, before, len(out))
    return out

def _drop_duplicate_rows(
    df: pd.DataFrame,
    columns: list[str],
    log_label: str,
) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=columns).copy()
    _log_row_count_change(log_label, before, len(out))
    return out

def _log_duplicate_pair_check(
    df: pd.DataFrame,
    columns: list[str],
    log_label: str,
) -> None:
    pair_counts = df.groupby(columns).size()
    duplicate_pair_counts = pair_counts[pair_counts > 1]
    logger.info(
        "%s: %d duplicate pairs involving %d rows",
        log_label,
        len(duplicate_pair_counts),
        int(duplicate_pair_counts.sum()),
    )

def _check_conflicting_prediction_scores(
    df: pd.DataFrame,
    query_column: str,
    target_column: str,
    prediction_column: str,
    duplicate_log_label: str,
) -> None:
    _log_duplicate_pair_check(
        df,
        [query_column, target_column],
        duplicate_log_label,
    )
    grouped = df.groupby([query_column, target_column])[prediction_column].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        raise ValueError("Conflicting prediction scores found for query-target pairs. ")

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
        header=None,
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
    
    cols = [
        raw_gene_ID_column,
        raw_gene_Name_column,
        raw_mirna_column,
        raw_prediction_column,
    ]
    df = _drop_invalid_rows(
        df,
        cols,
        "Drop invalid raw microT-CNN prediction rows",
    )
    df = _drop_duplicate_rows(
        df,
        cols,
        "Drop exact duplicate raw microT-CNN prediction rows",
    )
    _check_conflicting_prediction_scores(
        df,
        raw_mirna_column,
        raw_gene_ID_column,
        raw_prediction_column,
        "Check raw microT-CNN rows for duplicate miRNA-Ensembl Gene ID prediction pairs from raw column 1",
    )

    return df

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

def _drop_conflicting_and_duplicate_final_pairs(
    df: pd.DataFrame,
    ensembl_id_column: str,
    mimat_column: str,
    score_column: str,
) -> pd.DataFrame:
    before_conflict_drop = len(df)
    grouped = df.groupby([ensembl_id_column, mimat_column])[score_column].nunique()
    conflicts = grouped[grouped > 1]
    conflicting_pairs = set(conflicts.index)
    if conflicting_pairs:
        pair_index = list(zip(df[ensembl_id_column], df[mimat_column]))
        df = df.loc[[pair not in conflicting_pairs for pair in pair_index]].copy()
    _log_row_count_change(
        "Drop final Ensembl_ID-miRNA_ID pairs with conflicting scores after Ensembl mapping",
        before_conflict_drop,
        len(df),
    )

    before = len(df)
    out = df.drop_duplicates(subset=[ensembl_id_column, mimat_column, score_column]).copy()
    _log_row_count_change(
        "Drop duplicate final Ensembl_ID-miRNA_ID pairs with identical scores after Ensembl mapping",
        before,
        len(out),
    )
    return out

def build_output_table(
    df: pd.DataFrame,
    prediction_column: str,
    score_column: str,
    final_columns: list[str],
    ensembl_id_column: str,
    mimat_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[score_column] = pd.to_numeric(out[prediction_column], errors="coerce")
    out = out.loc[:, final_columns].copy()
    return _drop_conflicting_and_duplicate_final_pairs(
        out,
        ensembl_id_column,
        mimat_column,
        score_column,
    )
