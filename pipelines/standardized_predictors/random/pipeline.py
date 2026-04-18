#!/usr/bin/env python3
"""Generate deterministic random demo predictor TSVs."""

import argparse
import logging
from pathlib import Path

from funmirbench.build_predictions import (
    build_dataset_random_scores,
    build_random_scores,
    write_tsv,
)
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(parse_log_level(args.log_level))

    repo_root = Path(__file__).resolve().parents[3]
    experiments_tsv = repo_root / "metadata" / "mirna_experiment_info.tsv"
    out_dir = repo_root / "data" / "predictions" / "random"
    full_out_path = out_dir / "random_standardized.tsv"
    capped_out_path = out_dir / "random_3000_standardized.tsv"

    logger.info("Loading experiment registry for random predictors...")

    full_scores = build_random_scores(experiments_tsv, repo_root)
    logger.info("Writing full-coverage random predictor TSV to %s...", full_out_path)
    write_tsv(full_scores, full_out_path)
    logger.info("Wrote %s rows to %s", len(full_scores), full_out_path)

    capped_scores = build_dataset_random_scores(
        experiments_tsv,
        repo_root,
        max_genes_per_dataset=3000,
    )
    logger.info("Writing 3000-per-dataset random predictor TSV to %s...", capped_out_path)
    write_tsv(capped_scores, capped_out_path)
    logger.info("Wrote %s rows to %s", len(capped_scores), capped_out_path)


if __name__ == "__main__":
    main()
