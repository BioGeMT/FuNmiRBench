#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    add_standard_io_args,
    configure_file_logging,
    log_step,
    predictor_dir,
    repo_root,
    resolve_cli_path,
    write_standardized_table,
)
from utils import (  # noqa: E402
    build_output_table,
    create_mirna_name_to_mimat_mapping,
    download_file,
    load_predictions,
    map_mirna_names_to_mimat,
    resolve_path_relative_to_root,
)

logger = logging.getLogger("pipeline")


def parse_args(root: Path, pipeline_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize miRBind2 predictions for FuNmiRBench.")
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=pipeline_dir / "data" / "3utrs_mirbind2_predictions.tsv.gz",
        help="Raw miRBind2 TSV prediction file (downloaded from Zenodo by default)",
    )
    parser.add_argument(
        "--mirbase-mature",
        type=Path,
        default=root / "data" / "resources" / "mirbase" / "mature.fa",
        help="miRBase 22.1 mature.fa file",
    )
    add_standard_io_args(
        parser,
        default_output=root / "data" / "predictions" / "mirbind2" / "mirbind2_standardized.tsv",
        default_log_file=pipeline_dir / "mirbind2_pipeline.log",
    )
    return parser.parse_args()


def main() -> None:
    root = repo_root()
    pipeline_dir = predictor_dir("mirbind2", root=root)
    args = parse_args(root, pipeline_dir)
    args.predictions_file = resolve_cli_path(args.predictions_file, root)
    args.mirbase_mature = resolve_cli_path(args.mirbase_mature, root)
    args.output = resolve_cli_path(args.output, root)
    args.log_file = resolve_cli_path(args.log_file, root)

    configure_file_logging(args.log_file, args.log_level)
    logger.info("Starting miRBind2 standardization pipeline")

    raw_ensembl_column = "Gene_ID"
    raw_gene_name_column = "Gene_Symbol"
    raw_mirna_name_column = "miRNA_Name"
    raw_prediction_column = "miRBind2_3UTR_prediction"
    raw_columns = [
        "Transcript_ID",
        raw_ensembl_column,
        raw_gene_name_column,
        raw_mirna_name_column,
        raw_prediction_column,
    ]

    final_columns = ["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"]

    mirbind2_predictions_url = "https://zenodo.org/records/20609975/files/3utrs_mirbind2_predictions.tsv.gz?download=1"
    total_steps = 6

    log_step(logger, 1, total_steps, "Download or resolve raw miRBind2 predictions")
    raw_predictions_path = download_file(
        mirbind2_predictions_url,
        args.predictions_file,
        timeout=360,
        resource_label="miRBind2 raw prediction file",
    )

    log_step(logger, 2, total_steps, "Load raw miRBind2 predictions")
    pred_df = load_predictions(
        raw_predictions_path,
        required_columns=raw_columns,
        ensembl_column=raw_ensembl_column,
        gene_name_column=raw_gene_name_column,
        mirna_name_column=raw_mirna_name_column,
        score_column=raw_prediction_column,
    )

    log_step(logger, 3, total_steps, "Create miRNA name-to-MIMAT mapping from miRBase mature.fa")
    mirna_name_to_mimat = create_mirna_name_to_mimat_mapping(args.mirbase_mature)

    log_step(logger, 4, total_steps, "Annotate raw miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat,
        mirna_name_column=raw_mirna_name_column,
        mimat_column="miRNA_ID",
    )

    log_step(logger, 5, total_steps, "Build standardized output columns")
    final_df = build_output_table(
        pred_df,
        raw_ensembl_column=raw_ensembl_column,
        raw_gene_name_column=raw_gene_name_column,
        raw_mirna_name_column=raw_mirna_name_column,
        raw_prediction_column=raw_prediction_column,
        ensembl_column="Ensembl_ID",
        gene_name_column="Gene_Name",
        mimat_column="miRNA_ID",
        mirna_name_column="miRNA_Name",
        score_column="Score",
        final_columns=final_columns,
    )

    log_step(logger, 6, total_steps, "Validate and write standardized output table")
    write_standardized_table(final_df, args.output, logger=logger, columns=final_columns)
    logger.info("Relative output path: %s", resolve_path_relative_to_root(args.output))


if __name__ == "__main__":
    main()
