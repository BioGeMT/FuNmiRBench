#!/usr/bin/env python3
"""CLI entrypoint for the miRAW standardization pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests

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
    collapse_to_best_score_per_pair,
    create_ensembl_to_gene_name_mapping_from_gtf,
    create_mirna_name_to_mimat_mapping,
    download_file,
    load_miraw_predictions,
    map_ensembl_to_gene_name,
    map_mirna_names_to_mimat,
    resolve_path_relative_to_root,
)

logger = logging.getLogger("miraw_pipeline")

MIRAW_FIGSHARE_DOI = "10.6084/m9.figshare.32982218"
MIRAW_FIGSHARE_ARTICLE_ID = "32982218"
MIRAW_FIGSHARE_FILENAME = "best_per_pair.tsv.gz"


def resolve_figshare_download_url(article_id: str, filename: str, timeout: int = 120) -> str:
    metadata_url = f"https://api.figshare.com/v2/articles/{article_id}"
    try:
        response = requests.get(metadata_url, timeout=timeout)
        response.raise_for_status()
        metadata = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"Failed to resolve miRAW predictions from DOI {MIRAW_FIGSHARE_DOI}: {exc}"
        ) from exc

    for file_info in metadata.get("files", []):
        if file_info.get("name") == filename and file_info.get("download_url"):
            return str(file_info["download_url"])

    available_files = sorted(
        str(file_info.get("name"))
        for file_info in metadata.get("files", [])
        if file_info.get("name")
    )
    raise RuntimeError(
        f"Figshare record for DOI {MIRAW_FIGSHARE_DOI} does not contain {filename}. "
        f"Available files: {available_files}"
    )


def parse_args(root: Path, pipeline_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize miRAW predictions to the common FuNmiRBench schema"
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=pipeline_dir / "data" / MIRAW_FIGSHARE_FILENAME,
        help=(
            "Path to the preprocessed miRAW predictions file. When the default path is missing, "
            f"the file is downloaded from DOI {MIRAW_FIGSHARE_DOI}."
        ),
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=pipeline_dir / "data" / "resources",
        help="Directory where miRBase/Ensembl resources will be cached",
    )
    add_standard_io_args(
        parser,
        default_output=root / "data" / "predictions" / "miraw" / "miraw_standardized.tsv",
        default_log_file=pipeline_dir / "miraw_pipeline.log",
    )
    return parser.parse_args()


def main() -> None:
    root = repo_root()
    pipeline_dir = predictor_dir("miraw", root=root)
    args = parse_args(root, pipeline_dir)
    args.predictions_file = resolve_cli_path(args.predictions_file, root)
    args.resources_dir = resolve_cli_path(args.resources_dir, root)
    args.output = resolve_cli_path(args.output, root)
    args.log_file = resolve_cli_path(args.log_file, root)

    configure_file_logging(args.log_file, args.log_level)
    logger.info("Starting miRAW standardization pipeline")
    total_steps = 9

    log_step(logger, 1, total_steps, "Download or resolve preprocessed miRAW predictions")
    miraw_download_url = resolve_figshare_download_url(
        MIRAW_FIGSHARE_ARTICLE_ID,
        MIRAW_FIGSHARE_FILENAME,
    )
    predictions_path = download_file(
        miraw_download_url,
        args.predictions_file,
        timeout=360,
        resource_label=f"miRAW predictions from DOI {MIRAW_FIGSHARE_DOI}",
    )

    log_step(logger, 2, total_steps, "Resolve external miRBase/Ensembl resources")
    mirbase_url = "https://mirbase.org/download_version_files/22.1/mature.fa"
    mirbase_path = download_file(
        mirbase_url,
        args.resources_dir / "mirbase" / "mature.fa",
        resource_label="miRBase mature.fa resource",
    )

    ensembl_gtf_url = "https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz"
    ensembl_gtf_path = download_file(
        ensembl_gtf_url,
        args.resources_dir / "ensembl" / "Homo_sapiens.GRCh38.115.gtf.gz",
        timeout=360,
        resource_label="Ensembl v115 GTF resource",
    )

    log_step(logger, 3, total_steps, "Load preprocessed miRAW predictions")
    pred_df = load_miraw_predictions(predictions_path)

    log_step(logger, 4, total_steps, "Collapse to the best Prediction score per Ensembl_ID-miRNA pair")
    pred_df = collapse_to_best_score_per_pair(
        pred_df,
        ensembl_id_column="Ensembl_ID",
        mirna_name_column="miRNA_Name",
        score_column="Score",
    )

    log_step(logger, 5, total_steps, "Create miRNA name-to-MIMAT mapping from miRBase mature.fa")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)

    log_step(logger, 6, total_steps, "Create Ensembl gene ID-to-gene-name mapping from Ensembl v115 GTF")
    ensembl_to_gene_name_map = create_ensembl_to_gene_name_mapping_from_gtf(ensembl_gtf_path)

    log_step(logger, 7, total_steps, "Map miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat_map,
        mirna_name_column="miRNA_Name",
        mimat_column="miRNA_ID",
    )

    log_step(logger, 8, total_steps, "Map Ensembl IDs to gene names")
    pred_df = map_ensembl_to_gene_name(
        pred_df,
        ensembl_to_gene_name_map,
        ensembl_id_column="Ensembl_ID",
        gene_name_column="Gene_Name",
        raw_gene_name_column="Raw_Gene_Name",
    )

    log_step(logger, 9, total_steps, "Build, validate, and write standardized output table")
    final_columns = ["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"]
    final_df = build_output_table(
        pred_df,
        final_columns=final_columns,
        ensembl_id_column="Ensembl_ID",
        mimat_column="miRNA_ID",
        score_column="Score",
    )
    write_standardized_table(final_df, args.output, logger=logger, columns=final_columns)
    logger.info("Relative output path: %s", resolve_path_relative_to_root(args.output))


if __name__ == "__main__":
    main()
