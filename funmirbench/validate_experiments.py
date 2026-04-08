"""Validate that experiment DE tables referenced by metadata exist and are readable."""

import argparse
import logging
import pathlib
from typing import Iterable

import pandas as pd

from funmirbench.de_table import read_de_table
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)


def resolve_de_table_path(
    de_table_path: str | pathlib.Path,
    *,
    experiments_tsv: pathlib.Path,
    root: pathlib.Path | None = None,
) -> pathlib.Path:
    path = pathlib.Path(de_table_path)
    if path.is_absolute():
        return path

    if root is not None:
        return (root / path).resolve()

    candidates: Iterable[pathlib.Path] = (
        pathlib.Path.cwd(),
        experiments_tsv.parent,
    )
    for candidate_root in candidates:
        candidate = (candidate_root / path).resolve()
        if candidate.exists():
            return candidate
    return (pathlib.Path.cwd() / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Validate experiment DE table files.")
    parser.add_argument("--experiments-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(parse_log_level(args.log_level))

    experiments_tsv = args.experiments_tsv.resolve()
    root = args.root.resolve() if args.root is not None else None
    df = pd.read_csv(experiments_tsv, sep="\t")

    total = len(df)
    present = 0
    readable = 0
    missing = []
    unreadable = []

    for _, row in df.iterrows():
        path = resolve_de_table_path(
            row["de_table_path"],
            experiments_tsv=experiments_tsv,
            root=root,
        )
        if not path.exists():
            missing.append((row["id"], str(path)))
            continue
        present += 1
        try:
            read_de_table(path)
            readable += 1
        except Exception as exc:
            unreadable.append((row["id"], str(exc)))

    logger.info("Datasets in metadata: %s", total)
    logger.info("Files present:        %s", present)
    logger.info("Readable DE tables:   %s", readable)
    if missing:
        logger.warning("Missing files:        %s", len(missing))
        for dataset_id, path in missing[:20]:
            logger.warning("  - %s: %s", dataset_id, path)
    if unreadable:
        logger.error("Unreadable files:     %s", len(unreadable))
        for dataset_id, message in unreadable[:20]:
            logger.error("  - %s: %s", dataset_id, message)


if __name__ == "__main__":
    main()
