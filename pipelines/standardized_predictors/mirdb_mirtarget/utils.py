import logging
from pathlib import Path
from typing import Iterable, Optional

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

def download_file(
    url: str,
    output_path: Path,
    params: Optional[dict[str, str]] = None,
    timeout: int = 120,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("Using %s", resolve_path_relative_to_root(output_path))
        return output_path

    logger.info("Downloading %s", resolve_path_relative_to_root(output_path))

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    if not response.content.strip():
        raise RuntimeError(f"Empty response from {url}")
    response_text = response.content.decode("utf-8", errors="replace")
    if "Query ERROR:" in response_text or "BioMart::Exception" in response_text:
        raise RuntimeError(f"BioMart returned an error response from {url}")

    output_path.write_bytes(response.content)
    logger.info("Saved %s", resolve_path_relative_to_root(output_path))
    return output_path

def _drop_invalid_rows(
    df: pd.DataFrame,
    columns: list[str],
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

    logger.info("Keeping valid rows: %d/%d", len(out), before)
    return out

def _drop_duplicate_rows(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=columns).copy()
    logger.info("Keeping unique rows: %d/%d", len(out), before)
    return out

def _check_conflicting_prediction_scores(
    df: pd.DataFrame,
    query_column: str,
    target_column: str,
    prediction_column: str,
) -> None:
    grouped = df.groupby([query_column, target_column])[prediction_column].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        raise ValueError("Conflicting prediction scores found for query-target pairs. ")

def load_prediction_files(
    path: Path,
    raw_mirna_column: str,
    raw_transcript_column: str,
    raw_prediction_column: str,
    raw_ncbi_gene_id_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=[
            raw_mirna_column,
            raw_transcript_column,
            raw_prediction_column,
            raw_ncbi_gene_id_column,
        ],
        dtype=str,
    )
    if df.empty:
        raise RuntimeError("No prediction file was loaded")
    
    cols = [
        raw_mirna_column,
        raw_transcript_column,
        raw_prediction_column,
        raw_ncbi_gene_id_column,
    ]
    df = _drop_invalid_rows(df, cols)
    df = _drop_duplicate_rows(df, cols)
    _check_conflicting_prediction_scores(
        df,
        raw_mirna_column,
        raw_transcript_column,
        raw_prediction_column,
    )
    _check_conflicting_prediction_scores(
        df,
        raw_mirna_column,
        raw_ncbi_gene_id_column,
        raw_prediction_column,
    )

    return df

def _drop_conflicting_refseq_rows(
    df: pd.DataFrame,
    refseq_column: str,
    ensembl_id_column: str,
    gene_name_column: str,
) -> pd.DataFrame:
    before = len(df)

    counts = df.groupby(refseq_column).size()
    conflicting_refseq_ids = counts[counts > 1].index

    out = df.loc[~df[refseq_column].isin(conflicting_refseq_ids)].copy()

    logger.info("Keeping non-conflicting rows: %d/%d", len(out), before)
    return out

def create_refseq_to_ensembl_mapping(
    biomart_path: Path,
    biomart_ensembl_id_column: str,
    biomart_gene_name_column: str,
    biomart_refseq_column: str,
) -> dict[str, tuple[str, str]]:
    biomart = pd.read_csv(biomart_path, sep="\t", dtype=str)
    required_columns = {biomart_ensembl_id_column, biomart_gene_name_column, biomart_refseq_column}
    missing = required_columns - set(biomart.columns)
    if missing:
        raise ValueError(f"{biomart_path} is missing columns: {missing}")

    biomart_columns = [
        biomart_ensembl_id_column,
        biomart_gene_name_column,
        biomart_refseq_column,
    ]
    biomart = _drop_invalid_rows(
        biomart.loc[:, biomart_columns].copy(),
        biomart_columns,
    )
    biomart = _drop_duplicate_rows(
        biomart,
        biomart_columns,
    )
    biomart = _drop_conflicting_refseq_rows(
        biomart,
        biomart_refseq_column,
        biomart_ensembl_id_column,
        biomart_gene_name_column,
    )

    mapping: dict[str, tuple[str, str]] = {}
    for _, row in biomart.iterrows():
        refseq_id = str(row[biomart_refseq_column]).strip()
        mapping[refseq_id] = (
            row[biomart_ensembl_id_column],
            row[biomart_gene_name_column],
        )

    return mapping

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
) -> pd.DataFrame:
    before = len(df)
    out = df.loc[df[mapped_column].notna()].copy()

    logger.info(
        "Mapped rows: %d/%d",
        len(out),
        before,
    )

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
    return _drop_unmapped_rows(out, mimat_column)

def map_refseq_to_ensembl(
    df: pd.DataFrame,
    refseq_to_ensembl_map: dict[str, tuple[str, str]],
    refseq_column: str,
    ensembl_id_column: str,
    gene_name_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[refseq_column] = out[refseq_column].astype(str).str.strip()
    mapped = out[refseq_column].map(refseq_to_ensembl_map)
    out[ensembl_id_column] = mapped.str[0]
    out[gene_name_column] = mapped.str[1]
    return _drop_unmapped_rows(out, ensembl_id_column)

def _drop_conflicting_ncbi_gene_id_rows(
    df: pd.DataFrame,
    ncbi_gene_id_column: str,
    ensembl_id_column: str,
    gene_name_column: str,
) -> pd.DataFrame:
    before = len(df)

    counts = df.groupby(ncbi_gene_id_column).size()
    conflicting_ncbi_gene_ids = counts[counts > 1].index

    out = df.loc[~df[ncbi_gene_id_column].isin(conflicting_ncbi_gene_ids)].copy()

    logger.info("Keeping non-conflicting rows: %d/%d", len(out), before)
    return out

def create_ncbi_gene_id_to_ensembl_mapping(
    biomart_path: Path,
    biomart_ensembl_id_column: str,
    biomart_gene_name_column: str,
    biomart_ncbi_gene_id_column: str,
) -> dict[str, tuple[str, str]]:
    biomart = pd.read_csv(
        biomart_path,
        sep="\t",
        dtype=str,
    )
    required_columns = {
        biomart_ensembl_id_column,
        biomart_gene_name_column,
        biomart_ncbi_gene_id_column,
    }
    missing = required_columns - set(biomart.columns)
    if missing:
        raise ValueError(f"{biomart_path} is missing columns: {missing}")

    biomart_columns = [
        biomart_ensembl_id_column,
        biomart_gene_name_column,
        biomart_ncbi_gene_id_column,
    ]
    biomart = _drop_invalid_rows(
        biomart.loc[:, biomart_columns].copy(),
        biomart_columns,
    )
    biomart = _drop_duplicate_rows(
        biomart,
        biomart_columns,
    )
    biomart = _drop_conflicting_ncbi_gene_id_rows(
        biomart,
        biomart_ncbi_gene_id_column,
        biomart_ensembl_id_column,
        biomart_gene_name_column,
    )

    mapping: dict[str, tuple[str, str]] = {}
    for _, row in biomart.iterrows():
        ncbi_gene_id = str(row[biomart_ncbi_gene_id_column]).strip()
        mapping[ncbi_gene_id] = (
            row[biomart_ensembl_id_column],
            row[biomart_gene_name_column],
        )

    return mapping

def map_ncbi_gene_id_to_ensembl(
    df: pd.DataFrame,
    ncbi_gene_id_to_ensembl_map: dict[str, tuple[str, str]],
    ncbi_gene_id_column: str,
    ensembl_id_column: str,
    gene_name_column: str,
    *,
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out[ncbi_gene_id_column] = out[ncbi_gene_id_column].astype(str).str.strip()
    mapped = out[ncbi_gene_id_column].map(ncbi_gene_id_to_ensembl_map)
    out[ensembl_id_column] = mapped.str[0]
    out[gene_name_column] = mapped.str[1]
    if drop_unmapped:
        return _drop_unmapped_rows(out, ensembl_id_column)

    logger.info(
        "Mapped rows: %d/%d",
        out[ensembl_id_column].notna().sum(),
        len(out),
    )
    return out

def fill_unmapped_rows_with_refseq_to_ensembl(
    df: pd.DataFrame,
    refseq_to_ensembl_map: dict[str, tuple[str, str]],
    refseq_column: str,
    ensembl_id_column: str,
    gene_name_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[refseq_column] = out[refseq_column].astype(str).str.strip()

    needs_mapping = out[ensembl_id_column].isna()
    before_unmapped = int(needs_mapping.sum())

    mapped = out.loc[needs_mapping, refseq_column].map(refseq_to_ensembl_map)
    out.loc[needs_mapping, ensembl_id_column] = mapped.str[0]
    out.loc[needs_mapping, gene_name_column] = mapped.str[1]

    after_unmapped = int(out[ensembl_id_column].isna().sum())
    logger.info(
        "Fallback RefSeq mapping rescued rows: %d/%d",
        before_unmapped - after_unmapped,
        before_unmapped,
    )
    return _drop_unmapped_rows(out, ensembl_id_column)

def _collapse_final_pairs(
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
    logger.info(
        "Rows after dropping conflicting Ensembl ID : miRNA MIMAT pairs: %d/%d",
        len(df),
        before_conflict_drop,
    )

    before_dedup = len(df)
    out = df.drop_duplicates(
        subset=[ensembl_id_column, mimat_column, score_column],
    ).copy()
    logger.info(
        "Final rows after exact deduplication on Ensembl ID : miRNA MIMAT : score: %d/%d",
        len(out),
        before_dedup,
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
    return _collapse_final_pairs(
        out,
        ensembl_id_column,
        mimat_column,
        score_column,
    )
