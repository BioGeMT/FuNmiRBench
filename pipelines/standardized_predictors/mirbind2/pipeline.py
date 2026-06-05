#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from utils import (
    build_output_table,
    configure_logging,
    create_mirna_name_to_mimat_mapping,
    load_predictions,
    map_mirna_names_to_mimat,
    repo_root,
    resolve_path_relative_to_root,
)

logger = logging.getLogger("pipeline")


def log_step(step_number: int, total_steps: int, message: str) -> None:
    logger.info("Step %d/%d: %s", step_number, total_steps, message)


def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    root = repo_root()
    pipeline_dir = root / "pipelines" / "standardized_predictors" / "mirbind2"

    parser = argparse.ArgumentParser(description="Standardize miRBind2 predictions for FuNmiRBench.")
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=pipeline_dir / "unified_human_mirbase_mane_select_3utrs_selected_18_mirnas_mirbind2_predictions.tsv",
        help="Raw miRBind2 TSV prediction file",
    )
    parser.add_argument(
        "--mirbase-mature",
        type=Path,
        default=root / "data" / "resources" / "mirbase" / "mature.fa",
        help="miRBase 22.1 mature.fa file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "predictions" / "mirbind2" / "mirbind2_standardized.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=pipeline_dir / "mirbind2_pipeline.log",
        help="Log file path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Default: INFO",
    )

    args = parser.parse_args()
    args.predictions_file = resolve_cli_path(args.predictions_file, root)
    args.mirbase_mature = resolve_cli_path(args.mirbase_mature, root)
    args.output = resolve_cli_path(args.output, root)
    args.log_file = resolve_cli_path(args.log_file, root)

    configure_logging(args.log_file, args.log_level)
    logger.info("Starting miRBind2 pipeline")

    raw_ensembl_column = "Gene_ID"
    raw_gene_name_column = "Gene_Symbol"
    raw_mirna_name_column = "miRNA_Name"
    raw_score_column = "miRBind2_3UTR_prediction"
    raw_columns = [
        "Transcript_ID",
        raw_ensembl_column,
        raw_gene_name_column,
        raw_mirna_name_column,
        "miRNA_Sequence",
        "UTR_sequence",
        raw_score_column,
    ]

    mimat_column = "miRNA_ID"
    ensembl_column = "Ensembl_ID"
    gene_name_column = "Gene_Name"
    mirna_name_column = "miRNA_Name"
    score_column = "Score"

    total_steps = 5
    log_step(1, total_steps, "Load raw miRBind2 predictions")
    pred_df = load_predictions(
        args.predictions_file,
        required_columns=raw_columns,
        ensembl_column=raw_ensembl_column,
        gene_name_column=raw_gene_name_column,
        mirna_name_column=raw_mirna_name_column,
        score_column=raw_score_column,
    )

    log_step(2, total_steps, "Create miRNA name-to-MIMAT mapping from miRBase mature.fa")
    mirna_name_to_mimat = create_mirna_name_to_mimat_mapping(args.mirbase_mature)

    log_step(3, total_steps, "Annotate raw miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat,
        mirna_name_column=raw_mirna_name_column,
        mimat_column=mimat_column,
    )

    log_step(4, total_steps, "Drop transcript and sequence columns, then build standardized schema")
    final_df = build_output_table(
        pred_df,
        raw_ensembl_column=raw_ensembl_column,
        raw_gene_name_column=raw_gene_name_column,
        raw_mirna_name_column=raw_mirna_name_column,
        raw_score_column=raw_score_column,
        ensembl_column=ensembl_column,
        gene_name_column=gene_name_column,
        mimat_column=mimat_column,
        mirna_name_column=mirna_name_column,
        score_column=score_column,
    )

    log_step(5, total_steps, "Write standardized output table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, sep="\t", index=False)
    logger.info("Output written to: %s", resolve_path_relative_to_root(args.output))
    logger.info("Rows written: %d", len(final_df))


if __name__ == "__main__":
    main()
