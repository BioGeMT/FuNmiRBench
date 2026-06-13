"""Shared helpers for standardized predictor pipeline entrypoints.

Keep predictor-specific parsing and biological mapping logic inside each predictor's
``utils.py``. This module is intentionally small: it standardizes repository path
resolution, logging setup, progress messages, and final standardized TSV writing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests


LOG_LEVEL_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
STANDARDIZED_COLUMNS = ["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"]
ZENODO_RECORD = "20557595"
ZENODO_API_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def repo_root() -> Path:
    """Return the repository root from this shared module location."""
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from funmirbench.logger import parse_log_level, setup_logging  # noqa: E402


def zenodo_download_url(filename: str | Path, *, record: str = ZENODO_RECORD) -> str:
    """Return the direct Zenodo API content URL for a file in the FuNmiRBench record."""
    safe_filename = quote(Path(filename).name, safe="")
    return f"https://zenodo.org/api/records/{record}/files/{safe_filename}/content"


def download_file_if_missing(
    url: str,
    output: Path,
    *,
    logger: logging.Logger | None = None,
    timeout: int = 120,
    resource_label: str = "file",
) -> Path:
    """Download ``url`` to ``output`` unless the file is already cached."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if logger is not None:
            logger.info("Using cached %s: %s", resource_label, output)
        return output

    if logger is not None:
        logger.info("Downloading %s: %s", resource_label, output)

    tmp_path = None
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty {resource_label} from {url}")
        tmp_path.replace(output)
        tmp_path = None
    finally:
        response.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    if logger is not None:
        logger.info("Saved %s: %s", resource_label, output)
    return output


def predictor_dir(tool_id: str, *, root: Path | None = None) -> Path:
    """Return ``pipelines/standardized_predictors/<tool_id>``."""
    return (root or ROOT) / "pipelines" / "standardized_predictors" / tool_id


def resolve_cli_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve CLI paths relative to the repository root unless absolute."""
    value = Path(path)
    if value.is_absolute():
        return value
    return (root or ROOT) / value


def add_log_level_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level. Default: INFO",
    )


def add_standard_io_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path,
    default_log_file: Path,
) -> None:
    """Add common output/log CLI arguments used by predictor standardizers."""
    parser.add_argument("--output", type=Path, default=default_output, help="Output standardized TSV path")
    parser.add_argument("--log-file", type=Path, default=default_log_file, help="Log file path")
    add_log_level_arg(parser)


def configure_file_logging(log_file: Path, log_level: str) -> None:
    """Configure console logging plus one predictor-specific file handler."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(parse_log_level(log_level))
    root_logger = logging.getLogger()

    # Avoid duplicate file handlers when tests call multiple pipelines in one process.
    log_file = log_file.resolve()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file:
            return

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)


def log_step(logger: logging.Logger, step_number: int, total_steps: int, message: str) -> None:
    logger.info("Step %d/%d: %s", step_number, total_steps, message)


def validate_standardized_table(
    df: Any,
    *,
    required_columns: Iterable[str] = STANDARDIZED_COLUMNS,
) -> None:
    """Validate the common standardized predictor output schema.

    The check is intentionally schema-only. Some predictors may legitimately have
    blank gene names or MIMAT IDs before future curation passes, so row-level QC
    should stay predictor-specific instead of blocking shared output writing.
    """
    required = list(required_columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Standardized predictor output is missing required columns: {missing}")


def write_standardized_table(
    df: Any,
    output: Path,
    *,
    logger: logging.Logger | None = None,
    columns: Iterable[str] = STANDARDIZED_COLUMNS,
) -> None:
    """Validate and write a standardized predictor TSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    validate_standardized_table(df, required_columns=columns)
    df.loc[:, columns].to_csv(output, sep="\t", index=False)
    if logger is not None:
        logger.info("Output written to: %s", output)
        logger.info("Rows written: %d", len(df))
