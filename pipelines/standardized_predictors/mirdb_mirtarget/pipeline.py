import argparse
import logging
from pathlib import Path
from utils import configure_logging, download_file, load_prediction_files, create_mirna_name_to_mimat_mapping, map_mirna_names_to_mimat, create_ncbi_gene_id_to_ensembl_mapping, map_ncbi_gene_id_to_ensembl, create_refseq_to_ensembl_mapping, fill_unmapped_rows_with_refseq_to_ensembl, build_output_table, repo_root, resolve_path_relative_to_root

logger = logging.getLogger("pipeline")

MIRDB_PREDICTIONS_URL = "https://mirdb.org/download/miRDB_v6.0_prediction_result_human_all_scores.txt.gz"

def main() -> None:
    root = repo_root()
    pipeline_dir = root / "pipelines" / "standardized_predictors" / "mirdb_mirtarget"
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", type=Path, default=pipeline_dir / "data" / "miRDB_v6.0_prediction_result_human_all_scores.txt.gz", help="Raw all-score predictions file from miRDB")
    parser.add_argument("--resources-dir", type=Path, default=pipeline_dir / "data" / "resources", help="Directory for downloaded miRBase/BioMart files")
    parser.add_argument("--output", type=Path, default=root / "data" / "predictions" / "mirdb_mirtarget" / "mirdb_mirtarget_standardised.tsv", help="Output TSV path")
    parser.add_argument("--log-file", type=Path, default=pipeline_dir / "mirdb_mirtarget.log", help="Log file path")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level. Default: INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] )

    args = parser.parse_args()
    configure_logging(args.log_file, args.log_level)
    logger.info("Starting pipeline")

    mirbase_url = "https://mirbase.org/download_version_files/22.1/mature.fa"
    mirbase_path = args.resources_dir / "mirbase" / "mature.fa"
    mirbase_path = download_file(mirbase_url, mirbase_path)

    biomart_query = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "1" count = "" datasetConfigVersion = "0.6" >
            <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
                <Attribute name = "ensembl_gene_id" />
                <Attribute name = "external_gene_name" />
                <Attribute name = "refseq_mrna" />
                <Attribute name = "entrezgene_id" />
            </Dataset>
        </Query>"""
    biomart_url = "https://Sep2025.archive.ensembl.org/biomart/martservice"
    biomart_path = args.resources_dir / "biomart" / "hsapiens_ncbi_gene_id_refseq_to_ensembl.tsv"
    biomart_path = download_file(biomart_url, biomart_path, params={"query": biomart_query}, timeout=360)

    raw_predictions_path = download_file(MIRDB_PREDICTIONS_URL, args.predictions_file, timeout=360)
    raw_mirna_column = "miRNA"
    raw_transcript_column = "refseq_id"
    raw_prediction_column = "prediction"
    raw_ncbi_gene_id_column = "ncbi_gene_id"
    logger.info("Loading predictions")
    pred_df = load_prediction_files(
        raw_predictions_path,
        raw_mirna_column,
        raw_transcript_column,
        raw_prediction_column,
        raw_ncbi_gene_id_column,
    )

    logger.info("Creating miRNA name to MIMAT ID mapping")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)

    logger.info("Creating NCBI Gene ID to Ensembl ID, Gene name mapping")
    biomart_ensembl_id_column = "Gene stable ID"
    biomart_gene_name_column = "Gene name"
    biomart_refseq_column = "RefSeq mRNA ID"
    biomart_ncbi_gene_id_column = "NCBI gene (formerly Entrezgene) ID"
    ncbi_gene_id_to_ensembl_map = create_ncbi_gene_id_to_ensembl_mapping(
        biomart_path,
        biomart_ensembl_id_column,
        biomart_gene_name_column,
        biomart_ncbi_gene_id_column,
    )
    logger.info("Creating RefSeq to Ensembl ID, Gene name mapping")
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

    logger.info("Mapping miRNA names to MIMAT IDs")
    pred_df = map_mirna_names_to_mimat(
        pred_df,
        mirna_name_to_mimat_map,
        raw_mirna_column,
        mirna_name_column,
        mimat_column,
    )

    logger.info("Mapping NCBI Gene IDs to Ensembl IDs and Gene names")
    pred_df = map_ncbi_gene_id_to_ensembl(
        pred_df,
        ncbi_gene_id_to_ensembl_map,
        raw_ncbi_gene_id_column,
        ensembl_id_column,
        gene_name_column,
        drop_unmapped=False,
    )
    logger.info("Falling back to RefSeq IDs for rows still missing Ensembl mappings")
    pred_df = fill_unmapped_rows_with_refseq_to_ensembl(
        pred_df,
        refseq_to_ensembl_map,
        raw_transcript_column,
        ensembl_id_column,
        gene_name_column,
    )
    logger.info("Building final schema output table")
    final_df = build_output_table(
        pred_df,
        raw_prediction_column,
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
