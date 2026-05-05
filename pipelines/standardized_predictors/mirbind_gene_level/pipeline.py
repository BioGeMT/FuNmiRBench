#!/usr/bin/env python3
"""Standardize miRBind2 gene-level predictions for FuNmiRBench."""

from __future__ import annotations

import argparse
import html
import logging
import pathlib
import re
import tempfile
from urllib.parse import unquote

import pandas as pd
import requests

from funmirbench.logger import parse_log_level, setup_logging


PREDICTOR_NAME = "mirbind_gene_level"
PREDICTION_FILE_ID = "TODO"
PREDICTION_FILE_NAME = "TODO.tsv"


logger = logging.getLogger(__name__)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def download_google_drive_file(file_id: str, destination: pathlib.Path, *, force: bool = False) -> None:
    if destination.exists() and not force:
        logger.info("Skipping %s (already exists)", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(
        "https://docs.google.com/uc",
        params={"export": "download", "id": file_id},
        stream=True,
        timeout=60,
    )

    if "content-disposition" not in response.headers:
        params = {"export": "download", "id": file_id}
        download_url = "https://docs.google.com/uc"
        form_match = re.search(
            r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
            response.text,
            flags=re.DOTALL,
        )
        if form_match:
            download_url = html.unescape(form_match.group(1))
            params = {}
            for input_tag in re.findall(r"<input[^>]+>", form_match.group(2)):
                name_match = re.search(r'name="([^"]+)"', input_tag)
                value_match = re.search(r'value="([^"]*)"', input_tag)
                if name_match and value_match:
                    params[html.unescape(name_match.group(1))] = html.unescape(value_match.group(1))
        else:
            token = next(
                (value for key, value in response.cookies.items() if key.startswith("download_warning")),
                None,
            )
            if token:
                params["confirm"] = token
            else:
                token_match = re.search(r"confirm=([0-9A-Za-z_-]+)", response.text)
                if token_match:
                    params["confirm"] = token_match.group(1)
                uuid_match = re.search(r'name="uuid" value="([^"]+)"', response.text)
                if uuid_match:
                    params["uuid"] = uuid_match.group(1)
        response = session.get(
            download_url,
            params=params,
            stream=True,
            timeout=60,
        )

    content_disposition = response.headers.get("Content-Disposition", "")
    filename_match = re.search(r"filename\*=UTF-8''(.+?)(?:;|$)", content_disposition)
    if filename_match:
        logger.info("Google Drive filename: %s", unquote(filename_match.group(1).strip()))
    else:
        filename_match = re.search(r'filename="(.+?)"', content_disposition)
        if filename_match:
            logger.info("Google Drive filename: %s", filename_match.group(1).strip())

    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type and "content-disposition" not in response.headers:
        raise RuntimeError(f"Google Drive returned an HTML page instead of file bytes for {file_id}.")

    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        tmp_path = pathlib.Path(handle.name)
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(destination)
    logger.info("Downloaded %s (%s bytes)", destination, destination.stat().st_size)


def build_standardized_predictions(predictions_path: pathlib.Path) -> pd.DataFrame:
    logger.info("Loading predictions: %s", predictions_path)
    df = pd.read_csv(predictions_path, sep="\t")

    required = {"Gene_ID", "Gene_Symbol", "miRNA_Name", "miRNA_Accession", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    standardized = pd.DataFrame({
        "Ensembl_ID": df["Gene_ID"].astype("string").str.strip(),
        "Gene_Name": df["Gene_Symbol"].astype("string").str.strip(),
        "miRNA_ID": df["miRNA_Accession"].astype("string").str.strip(),
        "miRNA_Name": df["miRNA_Name"].astype("string").str.strip(),
        "Score": pd.to_numeric(df["prediction"], errors="coerce"),
    })

    before_drop = len(standardized)
    standardized = standardized.dropna(subset=["Ensembl_ID", "miRNA_Name", "Score"])
    standardized = standardized[
        standardized["Ensembl_ID"].str.startswith("ENSG", na=False)
        & (standardized["miRNA_Name"] != "")
    ].copy()
    logger.info("Dropped %s rows with missing required values", f"{before_drop - len(standardized):,}")

    duplicate_rows = int(standardized.duplicated(["Ensembl_ID", "miRNA_Name"]).sum())
    if duplicate_rows:
        duplicate_pairs = int(standardized.groupby(["Ensembl_ID", "miRNA_Name"]).size().gt(1).sum())
        logger.info(
            "Collapsing %s duplicate rows across %s gene-miRNA pairs by highest prediction score",
            f"{duplicate_rows:,}",
            f"{duplicate_pairs:,}",
        )
        keep_idx = standardized.groupby(["Ensembl_ID", "miRNA_Name"])["Score"].idxmax()
        standardized = standardized.loc[keep_idx].copy()

    return standardized.sort_values(["Ensembl_ID", "miRNA_Name"]).reset_index(drop=True)[
        ["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize miRBind2 gene-level predictions for FuNmiRBench.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--force", action="store_true", help="Re-download cached inputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))

    root = repo_root()
    pipeline_dir = root / "pipelines" / "standardized_predictors" / PREDICTOR_NAME
    data_dir = pipeline_dir / "data"
    log_path = pipeline_dir / f"{PREDICTOR_NAME}_pipeline.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", log_path)

    predictions_path = data_dir / PREDICTION_FILE_NAME
    logger.info("Downloading/reusing miRBind2 predictions file")
    download_google_drive_file(PREDICTION_FILE_ID, predictions_path, force=args.force)

    standardized = build_standardized_predictions(predictions_path)

    out_dir = root / "data" / "predictions" / PREDICTOR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{PREDICTOR_NAME}_standardized.tsv"
    standardized.to_csv(out_path, sep="\t", index=False)

    logger.info(
        "%s -> wrote %s rows | %s unique genes | %s unique miRNAs | output=%s",
        PREDICTOR_NAME,
        f"{len(standardized):,}",
        f"{standardized['Ensembl_ID'].nunique():,}",
        f"{standardized['miRNA_Name'].nunique():,}",
        out_path,
    )


if __name__ == "__main__":
    main()
