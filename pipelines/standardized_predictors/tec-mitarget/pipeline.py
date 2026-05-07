import argparse
import logging
from pathlib import Path

from utils import (
    aggregate_mre_predictions_to_transcripts,
    build_output_table,
    configure_logging,
    create_mirna_name_to_mimat_mapping,
    create_refseq_to_ensembl_mapping,
    download_file,
    load_prediction_files,
    map_mirna_names_to_mimat,
    map_refseq_to_ensembl,
    repo_root,
    resolve_path_relative_to_root,
)

logger = logging.getLogger("pipeline")

def main() -> None:
    root = repo_root()
    pipeline_dir = root / "pipelines" / "standardized_predictors" / "tec-mitarget"
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, default=pipeline_dir / "data" / "TEC-miTarget-model-predictions", help="Root dir containing test_split_* folders")
    parser.add_argument("--resources-dir", type=Path, default=pipeline_dir / "data" / "resources", help="Directory for downloaded miRBase/BioMart files")
    parser.add_argument("--output", type=Path, default=root / "data" / "predictions" / "tec-mitarget" / "tec_mitarget_standardized.tsv", help="Output TSV path")
    parser.add_argument("--log-file", type=Path, default=pipeline_dir / "tec_mitarget_pipeline.log", help="Log file path")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level. Default: INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] )

    args = parser.parse_args()
    configure_logging(args.log_file, args.log_level)
    logger.info("Starting pipeline")

    mirbase_url = "https://mirbase.org/download_version_files/22.1/mature.fa"
    mirbase_path = args.resources_dir / "mirbase" / "mature.fa"
    mirbase_path = download_file(
        mirbase_url,
        mirbase_path,
        resource_label="miRBase mature.fa resource",
    )

    biomart_query = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
            <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
                <Attribute name = "ensembl_gene_id" />
                <Attribute name = "external_gene_name" />
                <Attribute name = "refseq_mrna" />
            </Dataset>
        </Query>"""
    biomart_url = "https://Sep2025.archive.ensembl.org/biomart/martservice"
    biomart_path = args.resources_dir / "biomart" / "hsapiens_refseq_to_ensembl.tsv"
    biomart_path = download_file(
        biomart_url,
        biomart_path,
        params={"query": biomart_query},
        timeout=300,
        resource_label="BioMart RefSeq-to-Ensembl mapping table",
    )

    splits = range(0, 10)
    mre_prediction_file_name = "predict.tsv"
    query_column = "query_ids"
    target_column = "target_ids"
    prediction_column = "predictions"
    prediction_paths = [
        args.predictions_root / f"test_split_{split}" / mre_prediction_file_name
        for split in splits
    ]
    logger.info("Loading MRE-level predictions from %s", mre_prediction_file_name)
    pred_df = load_prediction_files(
        prediction_paths,
        query_column,
        target_column,
        prediction_column,
    )

    logger.info("Creating transcript-level predictions using max aggregation")
    pred_df = aggregate_mre_predictions_to_transcripts(
        pred_df,
        query_column,
        target_column,
        prediction_column,
    )

    logger.info("Creating miRNA name-to-MIMAT mapping from miRBase mature.fa")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)

    logger.info("Creating RefSeq-to-Ensembl gene mapping from BioMart table")
    biomart_ensembl_id_column = "Gene stable ID"
    biomart_gene_name_column = "Gene name"
    biomart_refseq_column = "RefSeq mRNA ID"
    refseq_to_ensembl_map = create_refseq_to_ensembl_mapping(
        biomart_path,
        biomart_ensembl_id_column,
        biomart_gene_name_column,
        biomart_refseq_column,
    )

    # Define final schema
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

    logger.info("Mapping prediction miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat_map,
        query_column,
        mirna_name_column,
        mimat_column,
    )

    logger.info("Mapping prediction RefSeq transcript IDs to Ensembl gene IDs and gene names")
    pred_df = map_refseq_to_ensembl(
        pred_df,
        refseq_to_ensembl_map,
        target_column,
        ensembl_id_column,
        gene_name_column,
    )
    logger.info("Building final standardized output table")
    final_df = build_output_table(
        pred_df,
        prediction_column,
        score_column,
        final_columns,
        ensembl_id_column,
        mimat_column,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, sep="\t", index=False)
    logger.info("Output written to: %s", resolve_path_relative_to_root(args.output))


if __name__ == "__main__":
    main()
