#!/usr/bin/env python3
"""Generate the perfect demo predictor TSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_log_level_arg, log_step, repo_root  # noqa: E402
from funmirbench.build_cheating_predictions import DEFAULT_ABS_LOGFC_THRESHOLD, DEFAULT_FDR_THRESHOLD  # noqa: E402
from funmirbench.build_perfect_predictions import DEMO_DATASET_IDS, build_perfect_scores  # noqa: E402
from funmirbench.build_predictions import write_tsv  # noqa: E402
from funmirbench.logger import parse_log_level, setup_logging  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the perfect demo predictor TSV.")
    add_log_level_arg(parser)
    return parser.parse_args()


def write_threshold_metadata(metadata_path: Path) -> None:
    metadata_path.write_text(
        json.dumps(
            {
                "tool_id": "perfect",
                "fdr_threshold": DEFAULT_FDR_THRESHOLD,
                "abs_logfc_threshold": DEFAULT_ABS_LOGFC_THRESHOLD,
                "dataset_ids": DEMO_DATASET_IDS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))

    root = repo_root()
    experiments_tsv = root / "metadata" / "mirna_experiment_info.tsv"
    out_path = root / "data" / "predictions" / "perfect" / "perfect_standardized.tsv"
    metadata_path = out_path.with_suffix(out_path.suffix + ".meta.json")

    total_steps = 2
    log_step(logger, 1, total_steps, "Build and write perfect predictor TSV")
    scores = build_perfect_scores(experiments_tsv, root)
    write_tsv(scores, out_path)
    logger.info("Wrote %s rows to %s", len(scores), out_path)

    log_step(logger, 2, total_steps, "Write threshold metadata sidecar")
    write_threshold_metadata(metadata_path)
    logger.info("Wrote predictor metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
