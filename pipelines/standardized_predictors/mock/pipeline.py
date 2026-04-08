#!/usr/bin/env python3
"""Generate the weak demo predictor TSV."""

import argparse
import logging
from pathlib import Path

from funmirbench.build_predictions import build_mock_scores, write_tsv
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(parse_log_level(args.log_level))

    repo_root = Path(__file__).resolve().parents[3]
    experiments_tsv = repo_root / "metadata" / "mirna_experiment_info.tsv"
    out_path = repo_root / "data" / "predictions" / "mock" / "mock_standardized.tsv"
    logger.info("Loading experiment registry for mock predictor...")
    scores = build_mock_scores(experiments_tsv, repo_root)
    logger.info("Writing mock predictor TSV to %s...", out_path)
    write_tsv(scores, out_path)
    logger.info("Wrote %s rows to %s", len(scores), out_path)


if __name__ == "__main__":
    main()
