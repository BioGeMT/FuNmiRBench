import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from funmirbench.logger import (
    DEFAULT_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    parse_log_level,
    setup_logging,
)

logger = logging.getLogger("utils")

MIRBASE_RELEASE = "22.1"


def repo_root() -> Path:
    return ROOT


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


def _drop_invalid_rows(
    df: pd.DataFrame,
    columns: list[str],
    log_label: str,
) -> pd.DataFrame:
    before = len(df)
    subset = df[columns]
    normalized = subset.astype("string").apply(lambda col: col.str.strip().str.lower())
    valid_rows = (
        ~subset.isna().any(axis=1)
        & ~(normalized == "").any(axis=1)
        & ~(normalized == "nan").any(axis=1)
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


def _drop_unmapped_rows(
    df: pd.DataFrame,
    mapped_column: str,
    log_label: str,
) -> pd.DataFrame:
    before = len(df)
    out = df.loc[df[mapped_column].notna()].copy()
    _log_row_count_change(log_label, before, len(out))
    return out


def _check_and_deduplicate_final_pairs(
    df: pd.DataFrame,
    ensembl_column: str,
    mimat_column: str,
    score_column: str,
) -> pd.DataFrame:
    grouped = df.groupby([ensembl_column, mimat_column])[score_column].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        sample_pairs = ", ".join(
            f"{ensembl_id}/{mimat_id}" for ensembl_id, mimat_id in conflicts.index[:5]
        )
        raise ValueError(
            "Conflicting scores found for standardized Ensembl_ID/miRNA_ID pairs "
            f"after MIMAT mapping: {sample_pairs}"
        )

    before = len(df)
    out = df.drop_duplicates(subset=[ensembl_column, mimat_column]).copy()
    _log_row_count_change(
        "Drop duplicate standardized Ensembl_ID-miRNA_ID pairs after MIMAT mapping",
        before,
        len(out),
    )
    return out


def load_predictions(
    path: Path,
    *,
    required_columns: list[str],
    ensembl_column: str,
    gene_name_column: str,
    mirna_name_column: str,
    score_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df.empty:
        raise RuntimeError("No miRBind2 predictions were loaded")

    extra_columns = [column for column in df.columns if column not in required_columns]
    if extra_columns:
        logger.info("Raw miRBind2 input has extra columns that will be ignored: %s", extra_columns)

    df = _drop_invalid_rows(
        df,
        [ensembl_column, mirna_name_column, score_column],
        "Drop invalid raw miRBind2 prediction rows",
    )
    df = _drop_duplicate_rows(
        df,
        [ensembl_column, gene_name_column, mirna_name_column, score_column],
        "Drop exact duplicate raw miRBind2 prediction rows",
    )
    return df


def create_mirna_name_to_mimat_mapping(mature_fa_path: Path) -> dict[str, str]:
    if not mature_fa_path.exists():
        raise FileNotFoundError(f"Missing miRBase mature.fa: {mature_fa_path}")

    mapping: dict[str, str] = {}
    with mature_fa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
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

    logger.info(
        "Loaded %d human miRNA name-to-MIMAT mappings from miRBase %s",
        len(mapping),
        MIRBASE_RELEASE,
    )
    return mapping


def map_mirna_names_to_mimat(
    df: pd.DataFrame,
    mirna_name_to_mimat: dict[str, str],
    *,
    mirna_name_column: str,
    mimat_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[mirna_name_column] = out[mirna_name_column].astype(str).str.strip()
    out[mimat_column] = out[mirna_name_column].map(mirna_name_to_mimat)
    return _drop_unmapped_rows(
        out,
        mimat_column,
        "Drop prediction rows with miRNA names that cannot map to MIMAT IDs",
    )


def build_output_table(
    df: pd.DataFrame,
    *,
    raw_ensembl_column: str,
    raw_gene_name_column: str,
    raw_mirna_name_column: str,
    raw_prediction_column: str,
    ensembl_column: str,
    gene_name_column: str,
    mimat_column: str,
    mirna_name_column: str,
    score_column: str,
    final_columns: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            ensembl_column: df[raw_ensembl_column].astype(str).str.strip(),
            gene_name_column: df[raw_gene_name_column].fillna("").astype(str).str.strip(),
            mimat_column: df[mimat_column].astype(str).str.strip(),
            mirna_name_column: df[raw_mirna_name_column].astype(str).str.strip(),
            score_column: pd.to_numeric(df[raw_prediction_column], errors="coerce"),
        }
    )
    out = out.loc[:, final_columns]
    out = _drop_invalid_rows(
        out,
        [ensembl_column, mimat_column, mirna_name_column, score_column],
        "Drop standardized rows with invalid canonical values or scores",
    )
    return _check_and_deduplicate_final_pairs(
        out,
        ensembl_column,
        mimat_column,
        score_column,
    )
