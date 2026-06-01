import gzip

from pipelines.standardized_predictors.microt_cnn.utils import (
    build_output_table,
    collapse_transcript_rows_to_genes,
    load_ensembl_tx_to_gene,
    load_prediction_files,
    map_transcripts_to_genes,
)


def test_microt_cnn_loader_skips_raw_header(tmp_path):
    raw_path = tmp_path / "microt.tsv"
    raw_path.write_text(
        "Ensembl_ID\tGene_Name\tmiRNA_Name\tScore\n"
        "ENST000001\tGENE1\thsa-miR-1\t0.7\n",
        encoding="utf-8",
    )

    df = load_prediction_files(
        raw_path,
        "Ensembl_ID",
        "Gene_Name",
        "miRNA_Name",
        "Score",
    )

    assert len(df) == 1
    assert df.iloc[0]["Ensembl_ID"] == "ENST000001"


def test_microt_cnn_maps_transcripts_to_gene_ids_and_collapses_duplicates(tmp_path):
    tx2gene_path = tmp_path / "tx2gene.tsv.gz"
    with gzip.open(tx2gene_path, "wt", encoding="utf-8") as handle:
        handle.write("transcript_id\tgene_id\n")
        handle.write("ENST000001.1\tENSG000001.5\n")
        handle.write("ENST000002.1\tENSG000001.5\n")
        handle.write("ENST000003.1\tENSG000002.1\n")

    tx_to_gene = load_ensembl_tx_to_gene(tx2gene_path)

    raw_path = tmp_path / "microt.tsv"
    raw_path.write_text(
        "Ensembl_ID\tGene_Name\tmiRNA_Name\tScore\n"
        "ENST000001\tGENE1\thsa-miR-1\t0.7\n"
        "ENST000002\tGENE1\thsa-miR-1\t0.8\n"
        "ENST000003\tGENE2\thsa-miR-1\t0.4\n"
        "ENST_MISSING\tGENE3\thsa-miR-1\t0.9\n",
        encoding="utf-8",
    )
    df = load_prediction_files(
        raw_path,
        "Ensembl_ID",
        "Gene_Name",
        "miRNA_Name",
        "Score",
    )
    df["miRNA_ID"] = "MIMAT0000001"

    mapped = map_transcripts_to_genes(df, tx_to_gene, "Ensembl_ID", "Ensembl_ID")
    final = build_output_table(
        mapped,
        "Score",
        "Score",
        ["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"],
    )
    collapsed = collapse_transcript_rows_to_genes(
        final,
        "Ensembl_ID",
        "Gene_Name",
        "miRNA_Name",
        "miRNA_ID",
        "Score",
    )

    assert collapsed["Ensembl_ID"].tolist() == ["ENSG000001", "ENSG000002"]
    assert collapsed["Score"].tolist() == [0.8, 0.4]
