#!/usr/bin/env python3
"""Generate the strong demo-only predictor TSV."""

import argparse
import json
import logging
from pathlib import Path

from funmirbench.build_cheating_predictions import (
    DEFAULT_ABS_LOGFC_THRESHOLD,
    DEFAULT_FDR_THRESHOLD,
    DEMO_DATASET_IDS,
    build_cheating_scores,
)
from funmirbench.build_predictions import write_tsv
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(parse_log_level(args.log_level))

    repo_root = Path(__file__).resolve().parents[3]
    experiments_tsv = repo_root / "metadata" / "mirna_experiment_info.tsv"
    out_path = repo_root / "data" / "predictions" / "cheating" / "cheating_standardized.tsv"
    metadata_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    logger.info("Loading experiment registry for demo strong predictor...")
    scores = build_cheating_scores(
        experiments_tsv,
        repo_root,
        dataset_ids=DEMO_DATASET_IDS,
    )
    logger.info("Writing cheating predictor TSV to %s...", out_path)
    write_tsv(scores, out_path)
    metadata_path.write_text(
        json.dumps(
            {
                "tool_id": "cheating",
                "fdr_threshold": DEFAULT_FDR_THRESHOLD,
                "abs_logfc_threshold": DEFAULT_ABS_LOGFC_THRESHOLD,
                "dataset_ids": DEMO_DATASET_IDS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote predictor metadata to %s", metadata_path)
    logger.info("Wrote %s rows to %s", len(scores), out_path)


if __name__ == "__main__":
    main()
