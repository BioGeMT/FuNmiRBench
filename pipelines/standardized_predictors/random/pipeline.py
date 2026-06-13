#!/usr/bin/env python3
"""Generate deterministic random demo predictor TSVs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_log_level_arg, log_step, repo_root  # noqa: E402
from funmirbench.build_predictions import (  # noqa: E402
    build_dataset_random_scores,
    build_random_scores,
    write_tsv,
)
from funmirbench.logger import parse_log_level, setup_logging  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic random demo predictor TSVs.")
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))

    root = repo_root()
    experiments_tsv = root / "metadata" / "mirna_experiment_info.tsv"
    out_dir = root / "data" / "predictions" / "random"
    full_out_path = out_dir / "random_standardized.tsv"
    capped_out_path = out_dir / "random_3000_standardized.tsv"

    total_steps = 2
    log_step(logger, 1, total_steps, "Build and write full-coverage random predictor")
    full_scores = build_random_scores(experiments_tsv, root)
    write_tsv(full_scores, full_out_path)
    logger.info("Wrote %s rows to %s", len(full_scores), full_out_path)

    log_step(logger, 2, total_steps, "Build and write 3000-per-dataset random predictor")
    capped_scores = build_dataset_random_scores(
        experiments_tsv,
        root,
        max_genes_per_dataset=3000,
    )
    write_tsv(capped_scores, capped_out_path)
    logger.info("Wrote %s rows to %s", len(capped_scores), capped_out_path)


if __name__ == "__main__":
    main()
