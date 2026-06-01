import gzip
import ast
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("miraw_utils")

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path_relative_to_root(path: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root())
    except ValueError:
        return path


def configure_logging(log_path: Path, log_level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def _log_row_count_change(label: str, before: int, after: int) -> None:
    logger.info("%s: %d -> %d rows", label, before, after)


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
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    if not response.content.strip():
        raise RuntimeError(f"Failed to download {resource_label}: empty response from {url}")

    response_text = response.content.decode("utf-8", errors="replace")
    if "Query ERROR:" in response_text or "BioMart::Exception" in response_text:
        raise RuntimeError(f"Failed to download {resource_label}: BioMart returned an error")

    output_path.write_bytes(response.content)
    logger.info("Saved %s: %s", resource_label, relative_path)
    return output_path


def create_mirna_name_to_mimat_mapping(mature_fa_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}

    with open(mature_fa_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue

            parts = line[1:].strip().split()
            if len(parts) < 2:
                raise ValueError(
                    f"{mature_fa_path}: invalid FASTA header; {line.strip()}"
                )

            mirna_name, mimat_id = parts[0], parts[1]

            if not mirna_name.startswith("hsa-"):
                continue

            if not mimat_id.startswith("MIMAT"):
                raise ValueError(
                    f"{mature_fa_path}: invalid MIMAT ID; {line.strip()}"
                )

            if mirna_name in mapping and mapping[mirna_name] != mimat_id:
                raise ValueError(
                    f"{mature_fa_path}: conflicting MIMAT for {mirna_name}: "
                    f"{mapping[mirna_name]} vs {mimat_id}"
                )

            mapping[mirna_name] = mimat_id

    logger.info("Loaded %d human miRNA->MIMAT mappings", len(mapping))
    return mapping


def _strip_ensembl_version(value: object) -> str:
    return str(value).strip().split(".", 1)[0]


def _parse_gtf_attributes(raw_attrs: str) -> dict[str, str]:
    attrs = {}
    for part in [item.strip() for item in raw_attrs.strip().split(";") if item.strip()]:
        if " " not in part:
            continue
        key, value = part.split(" ", 1)
        attrs[key] = value.strip().strip('"')
    return attrs


def create_ensembl_to_gene_name_mapping_from_gtf(gtf_gz_path: Path) -> dict[str, str]:
    if not gtf_gz_path.exists():
        raise FileNotFoundError(f"Missing Ensembl GTF file: {gtf_gz_path}")

    mapping: dict[str, str] = {}
    n_gene_rows = 0
    n_conflicting = 0
    with gzip.open(gtf_gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            n_gene_rows += 1
            attrs = _parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id")
            gene_name = attrs.get("gene_name")
            if not gene_id or not gene_name:
                continue
            gene_id = _strip_ensembl_version(gene_id)
            gene_name = str(gene_name).strip()
            if not gene_id or not gene_name:
                continue
            if gene_id in mapping and mapping[gene_id] != gene_name:
                n_conflicting += 1
                continue
            mapping[gene_id] = gene_name

    logger.info(
        "Loaded %d Ensembl v115 gene-name mappings from %d GTF gene rows; conflicts=%d",
        len(mapping),
        n_gene_rows,
        n_conflicting,
    )
    return mapping


def _open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _extract_raw_gene_name_suffix(raw_gene_value: str) -> Optional[str]:
    if "__" not in raw_gene_value:
        return None
    suffix = raw_gene_value.split("__", 1)[1].strip()
    if not suffix or suffix.startswith("biotype="):
        return None
    return suffix

def _parse_miraw_line(line: str, line_no: int, predictions_path: Path) -> Optional[dict[str, object]]:
    try:
        record = ast.literal_eval(line)
    except (SyntaxError, ValueError) as exc:
        logger.debug(
            "Skipping unparsable miRAW line %d in %s: %s",
            line_no,
            predictions_path,
            exc,
        )
        return None

    if not isinstance(record, dict):
        logger.debug(
            "Skipping miRAW line %d in %s because it is not a dictionary",
            line_no,
            predictions_path,
        )
        return None

    required_keys = {"GeneName", "miRNA", "Prediction"}
    missing_keys = required_keys - set(record)
    if missing_keys:
        logger.debug(
            "Skipping miRAW line %d in %s because it is missing keys: %s",
            line_no,
            predictions_path,
            sorted(missing_keys),
        )
        return None

    raw_gene = str(record["GeneName"]).strip()
    mirna_name = str(record["miRNA"]).strip()
    score_raw = record["Prediction"]

    if "__" not in raw_gene:
        logger.debug(
            "Skipping miRAW line %d in %s because GeneName does not contain '__': %s",
            line_no,
            predictions_path,
            raw_gene,
        )
        return None

    ensembl_id = _strip_ensembl_version(raw_gene.split("__", 1)[0])

    return {
        "Raw_Gene_Field": raw_gene,
        "Ensembl_ID": ensembl_id,
        "Raw_Gene_Name": _extract_raw_gene_name_suffix(raw_gene),
        "miRNA_Name": mirna_name,
        "Score": score_raw,
    }


def _load_preprocessed_miraw_tsv(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        predictions_path,
        sep="\t",
        dtype={
            "Ensembl_ID": "string",
            "Raw_Gene_Name": "string",
            "miRNA_Name": "string",
        },
        keep_default_na=False,
    )
    required_columns = {"Ensembl_ID", "Raw_Gene_Name", "miRNA_Name", "Score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{predictions_path}: missing required preprocessed miRAW columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.loc[:, ["Ensembl_ID", "Raw_Gene_Name", "miRNA_Name", "Score"]].copy()
    logger.info("Loaded %d preprocessed miRAW TSV rows from %s", len(df), predictions_path)
    return _clean_loaded_miraw_predictions(df, duplicate_subset=["Ensembl_ID", "miRNA_Name", "Score"])


def _clean_loaded_miraw_predictions(
    df: pd.DataFrame,
    duplicate_subset: list[str],
) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["Ensembl_ID", "miRNA_Name", "Score"]).copy()
    df["Ensembl_ID"] = df["Ensembl_ID"].map(_strip_ensembl_version)
    df["Raw_Gene_Name"] = df["Raw_Gene_Name"].fillna("").astype(str).str.strip()
    df["miRNA_Name"] = df["miRNA_Name"].astype(str).str.strip()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score"])
    df = df[(df["Ensembl_ID"] != "") & (df["miRNA_Name"] != "")].copy()
    _log_row_count_change("Drop invalid parsed miRAW rows", before, len(df))

    before = len(df)
    df = df.drop_duplicates(subset=duplicate_subset).copy()
    _log_row_count_change("Drop exact duplicate parsed miRAW rows", before, len(df))
    return df


def _is_preprocessed_miraw_tsv(predictions_path: Path) -> bool:
    with _open_text_auto(predictions_path) as handle:
        first_line = handle.readline().rstrip("\n")
    return first_line.split("\t")[:4] == ["Ensembl_ID", "Raw_Gene_Name", "miRNA_Name", "Score"]


def load_miraw_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {predictions_path}")

    if _is_preprocessed_miraw_tsv(predictions_path):
        return _load_preprocessed_miraw_tsv(predictions_path)

    rows: list[dict[str, object]] = []
    bad_lines = 0
    with _open_text_auto(predictions_path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parsed_row = _parse_miraw_line(line, line_no, predictions_path)
            if parsed_row is None:
                bad_lines += 1
                continue

            rows.append(parsed_row)

    if not rows:
        raise RuntimeError("No miRAW prediction rows could be parsed from the input file")

    df = pd.DataFrame(rows)
    logger.info("Loaded %d parsed miRAW rows from %s", len(df), predictions_path)
    if bad_lines:
        logger.warning("Skipped %d unparsable input lines", bad_lines)

    return _clean_loaded_miraw_predictions(
        df,
        duplicate_subset=["Ensembl_ID", "miRNA_Name", "Score", "Raw_Gene_Field"],
    )


def collapse_to_best_score_per_pair(
    df: pd.DataFrame,
    ensembl_id_column: str,
    mirna_name_column: str,
    score_column: str,
) -> pd.DataFrame:
    before = len(df)
    sort_cols = [ensembl_id_column, mirna_name_column, score_column]
    out = (
        df.sort_values(sort_cols, ascending=[True, True, False])
        .drop_duplicates(subset=[ensembl_id_column, mirna_name_column], keep="first")
        .copy()
    )
    _log_row_count_change(
        "Collapse miRAW rows to best score per Ensembl_ID-miRNA_Name pair",
        before,
        len(out),
    )
    return out


def map_mirna_names_to_mimat(
    df: pd.DataFrame,
    mirna_name_to_id: dict[str, str],
    mirna_name_column: str,
    mimat_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[mimat_column] = out[mirna_name_column].map(mirna_name_to_id)
    before = len(out)
    out = out.dropna(subset=[mimat_column]).copy()
    _log_row_count_change(
        "Drop miRAW rows with miRNA names that cannot map to MIMAT IDs",
        before,
        len(out),
    )
    return out


def map_ensembl_to_gene_name(
    df: pd.DataFrame,
    ensembl_to_gene_name_map: dict[str, str],
    ensembl_id_column: str,
    gene_name_column: str,
    raw_gene_name_column: str,
) -> pd.DataFrame:
    out = df.copy()
    out[gene_name_column] = out[ensembl_id_column].map(ensembl_to_gene_name_map).astype("string")

    # Fallback to the raw suffix when it looks like an actual gene symbol.
    raw_gene_name = out[raw_gene_name_column].fillna("").astype("string").str.strip()
    needs_fallback = out[gene_name_column].isna() & (raw_gene_name != "")
    out.loc[needs_fallback, gene_name_column] = out.loc[needs_fallback, raw_gene_name_column].astype("string")

    missing_or_empty_after_fallback = out[gene_name_column].isna() | (
        out[gene_name_column].astype("string").str.strip() == ""
    )
    missing_after_fallback = int(missing_or_empty_after_fallback.sum())
    if missing_after_fallback:
        logger.info(
            "Gene names missing after Ensembl v115/raw suffix annotation: %d rows; "
            "using Ensembl_ID as Gene_Name placeholder",
            missing_after_fallback,
        )
        out.loc[missing_or_empty_after_fallback, gene_name_column] = out.loc[
            missing_or_empty_after_fallback, ensembl_id_column
        ].astype("string")

    before = len(out)
    out = out[out[gene_name_column].astype(str).str.strip() != ""].copy()
    _log_row_count_change(
        "Drop miRAW rows whose gene names remain empty after annotation",
        before,
        len(out),
    )
    return out


def build_output_table(
    df: pd.DataFrame,
    final_columns: list[str],
    ensembl_id_column: str,
    mimat_column: str,
    score_column: str,
) -> pd.DataFrame:
    out = df.loc[:, final_columns].copy()

    before = len(out)
    out = out.dropna(subset=[ensembl_id_column, mimat_column, score_column]).copy()
    for column in final_columns:
        if column != score_column:
            out[column] = out[column].astype(str).str.strip()
    out = out[
        (out[ensembl_id_column] != "")
        & (out[mimat_column] != "")
        & (out["miRNA_Name"] != "")
        & (out["Gene_Name"] != "")
    ].copy()
    _log_row_count_change("Drop invalid final miRAW rows", before, len(out))

    before = len(out)
    out = out.drop_duplicates(subset=[ensembl_id_column, mimat_column, score_column]).copy()
    _log_row_count_change("Drop exact duplicate final miRAW rows", before, len(out))

    pair_counts = out.groupby([ensembl_id_column, mimat_column])[score_column].nunique()
    conflicts = pair_counts[pair_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting final scores found for some Ensembl_ID-miRNA_ID pairs; "
            "the input may not have been collapsed to one best row per pair."
        )

    return out.sort_values([ensembl_id_column, mimat_column]).reset_index(drop=True)
