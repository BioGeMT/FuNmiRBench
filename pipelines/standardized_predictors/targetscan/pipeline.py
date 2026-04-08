#!/usr/bin/env python3
"""CLI entrypoint for the TargetScan standardization pipeline."""

from __future__ import annotations

import argparse
import logging
from utils import (
    compute_final_statistics,
    configure_logging,
    parse_mirbase_mature,
    repo_root,
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()

    log_file = root / "pipelines" / "standardized_predictors" / "targetscan" / "targetscan_pipeline.log"

    global logger
    logger = configure_logging(log_file, log_level=args.log_level)
    logger.info("Logging to file: %s", log_file)

    targetscan_dir = root / "pipelines" / "standardized_predictors" / "targetscan"
    data_dir = targetscan_dir / "data"
    out_predictions_dir = root / "data" / "predictions"

    files = step1_download_targetscan_files(data_dir, force=False)

    tx_index = step2_build_representative_transcript_index(files["Gene_info.txt"], species_id="9606")

    ensembl_gtf = step3_download_ensembl115_gtf(data_dir, force=False)
    ensembl_tables = step4_build_and_cache_ensembl115_tables(
        ensembl_gtf,
        cache_dir=data_dir,
        force_rebuild=False,
    )

    step5_qc_targetscan_vs_ensembl_transcripts(
        tx_index=tx_index,
        ensembl_tables=ensembl_tables,
    )

    mirbase_fa = step6_download_mirbase_mature(data_dir, force=False)
    mirbase_acc2name = parse_mirbase_mature(mirbase_fa)

    mirna_annotations = step_build_human_mirna_annotations(
        files["miR_Family_Info.txt"],
        mirbase_acc2name=mirbase_acc2name,
        species_id="9606",
    )

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
