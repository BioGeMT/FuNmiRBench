#!/usr/bin/env python3
"""CLI entrypoint for the microT-CNN standardization pipeline."""

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
    build_ensembl_tx_to_gene_from_gtf,
    build_output_table,
    collapse_transcript_rows_to_genes,
    create_mirna_name_to_mimat_mapping,
    download_file,
    load_ensembl_tx_to_gene,
    load_prediction_files,
    map_mirna_names_to_mimat,
    map_transcripts_to_genes,
    resolve_path_relative_to_root,
)

logger = logging.getLogger("pipeline")


def parse_args(root: Path, pipeline_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize microT-CNN predictions for FuNmiRBench.")
    parser.add_argument(
        "--tx2gene-file",
        type=Path,
        default=pipeline_dir / "data" / "resources" / "ensembl" / "ensembl115_tx2gene.tsv.gz",
        help="Ensembl transcript-to-gene mapping TSV, optionally gzipped",
    )
    parser.add_argument(
        "--ensembl-gtf-file",
        type=Path,
        default=pipeline_dir / "data" / "resources" / "ensembl" / "Homo_sapiens.GRCh38.115.gtf.gz",
        help="Ensembl v115 GTF used to build --tx2gene-file when the mapping cache is missing",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=pipeline_dir / "data" / "resources",
        help="Directory for downloaded miRBase files",
    )
    add_standard_io_args(
        parser,
        default_output=root / "data" / "predictions" / "microt_cnn" / "microt_cnn_standardized.tsv",
        default_log_file=pipeline_dir / "microt_cnn_pipeline.log",
    )
    return parser.parse_args()


def main() -> None:
    root = repo_root()
    pipeline_dir = predictor_dir("microt_cnn", root=root)
    args = parse_args(root, pipeline_dir)
    args.tx2gene_file = resolve_cli_path(args.tx2gene_file, root)
    args.ensembl_gtf_file = resolve_cli_path(args.ensembl_gtf_file, root)
    args.resources_dir = resolve_cli_path(args.resources_dir, root)
    args.output = resolve_cli_path(args.output, root)
    args.log_file = resolve_cli_path(args.log_file, root)

    configure_file_logging(args.log_file, args.log_level)
    logger.info("Starting microT-CNN standardization pipeline")
    total_steps = 7

    log_step(logger, 1, total_steps, "Resolve raw microT-CNN predictions and external resources")
    mirbase_url = "https://mirbase.org/download_version_files/22.1/mature.fa"
    mirbase_path = download_file(
        mirbase_url,
        args.resources_dir / "mirbase" / "mature.fa",
        resource_label="miRBase mature.fa resource",
    )

    microt_cnn_predictions_url = "10.5281/zenodo.20313523"
    raw_predictions_path = download_file(
        microt_cnn_predictions_url,
        pipeline_dir / "data" / "microT_CNN_prediction_result_human_all_scores_gene_level.tsv.gz",
        timeout=360,
        resource_label="microT-CNN all-score prediction file",
    )
    raw_gene_id_column = "Ensembl_ID"
    raw_gene_name_column = "Gene_Name"
    raw_mirna_column = "miRNA_Name"
    raw_prediction_column = "Score"

    log_step(logger, 2, total_steps, "Load raw microT-CNN predictions")
    pred_df = load_prediction_files(
        raw_predictions_path,
        raw_gene_id_column,
        raw_gene_name_column,
        raw_mirna_column,
        raw_prediction_column,
    )

    log_step(logger, 3, total_steps, "Create miRNA and transcript mapping resources")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)
    if args.tx2gene_file.exists():
        logger.info("Loading Ensembl transcript-to-gene mapping: %s", resolve_path_relative_to_root(args.tx2gene_file))
        tx_to_gene = load_ensembl_tx_to_gene(args.tx2gene_file)
    else:
        ensembl_gtf_url = "https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz"
        ensembl_gtf_path = download_file(
            ensembl_gtf_url,
            args.ensembl_gtf_file,
            timeout=360,
            resource_label="Ensembl v115 GTF resource",
        )
        tx_to_gene = build_ensembl_tx_to_gene_from_gtf(ensembl_gtf_path, args.tx2gene_file)

    mimat_column = "miRNA_ID"
    ensembl_id_column = "Ensembl_ID"
    gene_name_column = "Gene_Name"
    mirna_name_column = "miRNA_Name"
    score_column = "Score"
    final_columns = [ensembl_id_column, gene_name_column, mimat_column, mirna_name_column, score_column]

    log_step(logger, 4, total_steps, "Map prediction miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat_map,
        raw_mirna_column,
        mirna_name_column,
        mimat_column,
    )

    log_step(logger, 5, total_steps, "Map microT-CNN Ensembl transcript IDs to Ensembl gene IDs")
    pred_df = map_transcripts_to_genes(
        pred_df,
        tx_to_gene,
        raw_gene_id_column,
        ensembl_id_column,
    )

    log_step(logger, 6, total_steps, "Build standardized output columns")
    final_df = build_output_table(
        pred_df,
        raw_prediction_column,
        score_column,
        final_columns,
    )
    final_df = collapse_transcript_rows_to_genes(
        final_df,
        ensembl_id_column,
        gene_name_column,
        mirna_name_column,
        mimat_column,
        score_column,
    )

    log_step(logger, 7, total_steps, "Validate and write standardized output table")
    write_standardized_table(final_df, args.output, logger=logger, columns=final_columns)
    logger.info("Relative output path: %s", resolve_path_relative_to_root(args.output))


if __name__ == "__main__":
    main()
