import argparse
from pathlib import Path
from utils import setup_logging, download_file, load_prediction_files, create_mirna_name_to_mimat_mapping, map_mirna_names_to_mimat, create_refseq_to_ensembl_mapping, map_refseq_to_ensembl, build_output_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, default=Path("data/TEC-miTarget-model-predictions"), help="Root dir containing test_split_* folders")
    parser.add_argument("--resources-dir", type=Path, default=Path("data/resources"), help="Directory for downloaded miRBase/BioMart files")
    parser.add_argument("--output", type=Path, default=Path("tec_mitarget_standardised.tsv"), help="Output TSV path")
    parser.add_argument("--log-file", type=Path, default=Path("tec_mitarget.log"), help="Log file path")

    args = parser.parse_args()
    logger = setup_logging(args.log_file)
    logger.info("Starting pipeline")

    mirbase_url = "https://mirbase.org/download_version_files/22.1/mature.fa"
    mirbase_path = args.resources_dir / "mirbase" / "mature.fa"
    mirbase_path = download_file(mirbase_url, mirbase_path, logger)

    biomart_query = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
            <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
                <Attribute name = "ensembl_gene_id" />
                <Attribute name = "external_gene_name" />
                <Attribute name = "refseq_mrna" />
            </Dataset>
        </Query>"""
    biomart_url = "http://www.ensembl.org/biomart/martservice"
    biomart_path = args.resources_dir / "biomart" / "hsapiens_refseq_to_ensembl.tsv"
    biomart_path = download_file(biomart_url, biomart_path, logger, params={"query": biomart_query}, timeout=300)

    splits = range(0, 10)
    prediction_file_name = "predict-gene-level.tsv"
    query_column = "query_ids"
    target_column = "target_ids"
    prediction_column = "predictions"
    prediction_paths = [
        args.predictions_root / f"test_split_{split}" / prediction_file_name
        for split in splits
    ]
    logger.info("Loading predictions")
    pred = load_prediction_files(
        prediction_paths,
        query_column,
        target_column,
        prediction_column,
        logger,
    )

    logger.info("Creating miRNA name to MIMAT ID mapping")
    mirna_name_to_mimat_map = create_mirna_name_to_mimat_mapping(mirbase_path)

    logger.info("Creating RefSeq to Ensembl ID, Gene name mapping")
    biomart_ensembl_id_column = "Gene stable ID"
    biomart_gene_name_column = "Gene name"
    biomart_refseq_column = "RefSeq mRNA ID"
    refseq_to_ensembl_map = create_refseq_to_ensembl_mapping(
        biomart_path,
        biomart_ensembl_id_column,
        biomart_gene_name_column,
        biomart_refseq_column,
        logger,
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
    pred = map_mirna_names_to_mimat(
        pred,
        mirna_name_to_mimat_map,
        query_column,
        mirna_name_column,
        mimat_column,
        logger,
    )

    logger.info("Mapping RefSeq IDs to Ensembl IDs and Gene names")
    pred = map_refseq_to_ensembl(
        pred,
        refseq_to_ensembl_map,
        target_column,
        ensembl_id_column,
        gene_name_column,
        logger,
    )
    logger.info("Building final schema output table")
    final = build_output_table(
        pred,
        prediction_column,
        score_column,
        final_columns,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, sep="\t", index=False)
    logger.info("Output written to: %s", args.output)


if __name__ == "__main__":
    main()
