import argparse
import logging
from pathlib import Path
from utils import configure_logging, download_file, load_prediction_files, create_mirna_name_to_mimat_mapping, map_mirna_names_to_mimat, build_output_table, repo_root, resolve_path_relative_to_root

logger = logging.getLogger("pipeline")

def log_step(step_number: int, total_steps: int, message: str) -> None:
    logger.info("Step %d/%d: %s", step_number, total_steps, message)

def resolve_cli_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path

def main() -> None:
    root = repo_root()
    pipeline_dir = root / "pipelines" / "standardized_predictors" / "microt_cnn"
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", type=Path, default=pipeline_dir / "data" / "microT_CNN_prediction_result_human_all_scores_gene_level.tsv.gz", help="Raw all-score predictions file from microT-CNN")
    parser.add_argument("--resources-dir", type=Path, default=pipeline_dir / "data" / "resources", help="Directory for downloaded miRBase files")
    parser.add_argument("--output", type=Path, default=root / "data" / "predictions" / "microt_cnn" / "microt_cnn_standardized.tsv", help="Output TSV path")
    parser.add_argument("--log-file", type=Path, default=pipeline_dir / "microt_cnn_pipeline.log", help="Log file path")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level. Default: INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] )

    args = parser.parse_args()
    args.predictions_file = resolve_cli_path(args.predictions_file, root)
    args.resources_dir = resolve_cli_path(args.resources_dir, root)
    args.output = resolve_cli_path(args.output, root)
    args.log_file = resolve_cli_path(args.log_file, root)

    configure_logging(args.log_file, args.log_level)
    logger.info("Starting pipeline")
    total_steps = 5

    log_step(1, total_steps, "Resolve raw microT-CNN predictions and external miRBase resources")
    mirbase_url = "https://mirbase.org/download_version_files/22.1/mature.fa"
    mirbase_path = args.resources_dir / "mirbase" / "mature.fa"
    mirbase_path = download_file(
        mirbase_url,
        mirbase_path,
        resource_label="miRBase mature.fa resource",
    )

    microt_cnn_predictions_url = "10.5281/zenodo.20313523"
    raw_predictions_path = download_file(
        microt_cnn_predictions_url,
        args.predictions_file,
        timeout=360,
        resource_label="microT-CNN all-score prediction file",
    )
    raw_gene_ID_column = "Ensembl_ID"
    raw_gene_Name_column = "Gene_Name"
    raw_mirna_column = "miRNA_Name"
    raw_prediction_column = "Score"

    log_step(2, total_steps, "Load raw microT-CNN predictions")
    pred_df = load_prediction_files(
        raw_predictions_path,
        raw_gene_ID_column,
        raw_gene_Name_column,
        raw_mirna_column,
        raw_prediction_column,
    )

    log_step(3, total_steps, "Create miRNA name-to-MIMAT mapping from miRBase mature.fa")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)

    mimat_column = "miRNA_ID"
    ensembl_id_column = "Ensembl_ID"
    gene_name_column = "Gene_Name"
    mirna_name_column = "miRNA_Name"
    score_column = "Score"
    final_columns = [
        ensembl_id_column,
        gene_name_column,
        mimat_column,
        mirna_name_column,
        score_column,
    ]

    log_step(4, total_steps, "Map prediction miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat_map,
        raw_mirna_column,
        mirna_name_column,
        mimat_column,
    )

    log_step(5, total_steps, "Build and write final standardized output table")
    final_df = build_output_table(
        pred_df,
        raw_prediction_column,
        score_column,
        final_columns,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, sep="\t", index=False)
    logger.info("Output written to: %s", resolve_path_relative_to_root(args.output))


if __name__ == "__main__":
    main()
