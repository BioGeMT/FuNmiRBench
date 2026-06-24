#!/usr/bin/env python3
"""CLI entrypoint for the TargetScan standardization pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_log_level_arg, log_step, predictor_dir, repo_root  # noqa: E402
from utils import (  # noqa: E402
    compute_final_statistics,
    configure_logging,
    parse_mirbase_mature,
    step1_download_targetscan_files,
    step2_build_representative_transcript_index,
    step3_download_ensembl115_gtf,
    step4_build_and_cache_ensembl115_tables,
    step5_qc_targetscan_vs_ensembl_transcripts,
    step6_download_mirbase_mature,
    step_build_human_mirna_annotations,
    step_write_standardized_predictions,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize TargetScan predictions for FuNmiRBench.")
    add_log_level_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    targetscan_dir = predictor_dir("targetscan", root=root)
    log_file = targetscan_dir / "targetscan_pipeline.log"

    global logger
    logger = configure_logging(log_file, log_level=args.log_level)
    logger.info("Logging to file: %s", log_file)

    data_dir = targetscan_dir / "data"
    out_predictions_dir = root / "data" / "predictions"
    total_steps = 8

    log_step(logger, 1, total_steps, "Download and unzip TargetScan inputs")
    files = step1_download_targetscan_files(data_dir, force=False)

    log_step(logger, 2, total_steps, "Build representative-transcript index")
    tx_index = step2_build_representative_transcript_index(files["Gene_info.txt"], species_id="9606")

    log_step(logger, 3, total_steps, "Download Ensembl v115 GTF")
    ensembl_gtf = step3_download_ensembl115_gtf(data_dir, force=False)

    log_step(logger, 4, total_steps, "Build/cache Ensembl v115 tables")
    ensembl_tables = step4_build_and_cache_ensembl115_tables(
        ensembl_gtf,
        cache_dir=data_dir,
        force_rebuild=False,
    )

    log_step(logger, 5, total_steps, "Run TargetScan-vs-Ensembl transcript QC")
    step5_qc_targetscan_vs_ensembl_transcripts(
        tx_index=tx_index,
        ensembl_tables=ensembl_tables,
    )

    log_step(logger, 6, total_steps, "Download and parse miRBase mature annotations")
    mirbase_fa = step6_download_mirbase_mature(data_dir, force=False)
    mirbase_acc2name = parse_mirbase_mature(mirbase_fa)

    log_step(logger, 7, total_steps, "Build human miRNA annotations")
    mirna_annotations = step_build_human_mirna_annotations(
        files["miR_Family_Info.txt"],
        mirbase_acc2name=mirbase_acc2name,
        species_id="9606",
    )

    log_step(logger, 8, total_steps, "Write standardized TargetScan predictions")
    step_write_standardized_predictions(
        files["Summary_Counts.all_predictions.txt"],
        tx_index=tx_index,
        ensembl_tables=ensembl_tables,
        mirna_annotations=mirna_annotations,
        out_predictions_dir=out_predictions_dir,
        species_id="9606",
    )

    compute_final_statistics(out_predictions_dir)


if __name__ == "__main__":
    main()
