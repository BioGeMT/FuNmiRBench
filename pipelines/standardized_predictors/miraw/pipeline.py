import argparse
import logging
from pathlib import Path

from utils import (
    build_output_table,
    collapse_to_best_score_per_pair,
    configure_logging,
    create_ensembl_to_gene_name_mapping_from_gtf,
    create_mirna_name_to_mimat_mapping,
    download_file,
    load_miraw_predictions,
    map_ensembl_to_gene_name,
    map_mirna_names_to_mimat,
    repo_root,
    resolve_path_relative_to_root,
)

logger = logging.getLogger("miraw_pipeline")

def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def log_step(step_number: int, total_steps: int, message: str) -> None:
    logger.info("Step %d/%d: %s", step_number, total_steps, message)


def main() -> None:
    root = repo_root()
    pipeline_dir = root / "pipelines" / "standardized_predictors" / "miraw"

    parser = argparse.ArgumentParser(
        description="Standardize miRAW predictions to the common FunMiRBench schema"
    )

    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=pipeline_dir / "data" / "best_per_pair.tsv.gz",
        help="Path to the local preprocessed miRAW predictions file (.tsv, .tsv.gz, .txt, or .txt.gz). Default: pipelines/standardized_predictors/miraw/data/best_per_pair.tsv.gz",
    )

    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=pipeline_dir / "data" / "resources",
        help="Directory where miRBase/Ensembl resources will be cached",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "predictions" / "miraw" / "miraw_standardized.tsv",
        help="Output TSV path",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=pipeline_dir / "miraw_pipeline.log",
        help="Log file path",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    args.predictions_file = resolve_cli_path(args.predictions_file, root)
    args.resources_dir = resolve_cli_path(args.resources_dir, root)
    args.output = resolve_cli_path(args.output, root)
    args.log_file = resolve_cli_path(args.log_file, root)

    configure_logging(args.log_file, args.log_level)

    logger.info("Starting miRAW standardization pipeline")
    total_steps = 8

    log_step(1, total_steps, "Resolve external miRBase/Ensembl resources")
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

    log_step(2, total_steps, "Load preprocessed miRAW predictions")
    pred_df = load_miraw_predictions(args.predictions_file)

    log_step(3, total_steps, "Collapse to the best Prediction score per Ensembl_ID-miRNA pair")
    pred_df = collapse_to_best_score_per_pair(
        pred_df,
        ensembl_id_column="Ensembl_ID",
        mirna_name_column="miRNA_Name",
        score_column="Score",
    )

    log_step(4, total_steps, "Create miRNA name-to-MIMAT mapping from miRBase mature.fa")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)

    log_step(5, total_steps, "Create Ensembl gene ID-to-gene-name mapping from Ensembl v115 GTF")
    ensembl_to_gene_name_map = create_ensembl_to_gene_name_mapping_from_gtf(ensembl_gtf_path)

    log_step(6, total_steps, "Map miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat_map,
        mirna_name_column="miRNA_Name",
        mimat_column="miRNA_ID",
    )

    log_step(7, total_steps, "Map Ensembl IDs to gene names")
    pred_df = map_ensembl_to_gene_name(
        pred_df,
        ensembl_to_gene_name_map,
        ensembl_id_column="Ensembl_ID",
        gene_name_column="Gene_Name",
        raw_gene_name_column="Raw_Gene_Name",
    )

    log_step(8, total_steps, "Build and write final standardized output table")
    final_df = build_output_table(
        pred_df,
        final_columns=["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"],
        ensembl_id_column="Ensembl_ID",
        mimat_column="miRNA_ID",
        score_column="Score",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, sep="\t", index=False)
    logger.info("Output written to: %s", resolve_path_relative_to_root(args.output))


if __name__ == "__main__":
    main()
