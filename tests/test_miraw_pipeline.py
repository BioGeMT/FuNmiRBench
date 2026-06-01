import gzip
import os
import subprocess
from pathlib import Path

import pandas as pd

from pipelines.standardized_predictors.miraw.preprocess_miraw import (
    preprocess_miraw_predictions,
)
from pipelines.standardized_predictors.miraw.utils import (
    build_output_table,
    collapse_to_best_score_per_pair,
    create_ensembl_to_gene_name_mapping_from_gtf,
    load_miraw_predictions,
    map_ensembl_to_gene_name,
)


def test_preprocess_miraw_predictions_streams_key_order_and_keeps_best_score(tmp_path):
    raw = tmp_path / "all_ensg.txt.gz"
    out = tmp_path / "best_per_pair.tsv.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        handle.write(
            "{'miRNA': 'hsa-miR-1', 'Prediction': 0.2, "
            "'GeneName': 'ENSG000001.5__GENE1', 'Other': 'x'}\n"
        )
        handle.write(
            "{'GeneName': 'ENSG000001.5__GENE1', 'miRNA': 'hsa-miR-1', "
            "'Prediction': 0.9}\n"
        )
        handle.write(
            "{'GeneName': 'ENSG000002__GENE2', 'Extra': 1, "
            "'Prediction': 0.4, 'miRNA': 'hsa-miR-2'}\n"
        )

    stats = preprocess_miraw_predictions(raw, out)
    df = pd.read_csv(out, sep="\t")

    assert stats["rows_seen"] == 3
    assert stats["rows_parsed"] == 3
    assert stats["unique_pairs"] == 2
    assert df.to_dict("records") == [
        {
            "Ensembl_ID": "ENSG000001",
            "Raw_Gene_Name": "GENE1",
            "miRNA_Name": "hsa-miR-1",
            "Score": 0.9,
        },
        {
            "Ensembl_ID": "ENSG000002",
            "Raw_Gene_Name": "GENE2",
            "miRNA_Name": "hsa-miR-2",
            "Score": 0.4,
        },
    ]


def test_miraw_preprocessing_shell_script_keeps_best_score(tmp_path):
    raw = tmp_path / "all_ensgs.txt.gz"
    out = tmp_path / "best_per_pair.tsv.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        handle.write(
            "{'miRNA': 'hsa-miR-1', 'Prediction': 0.2, "
            "'GeneName': 'ENSG000001.5__GENE1'}\n"
        )
        handle.write(
            "{'GeneName': 'ENSG000001.5__GENE1', 'miRNA': 'hsa-miR-1', "
            "'Prediction': 0.9}\n"
        )
        handle.write(
            "{'GeneName': 'ENSG000002__biotype=protein_coding', "
            "'Prediction': 0.4, 'miRNA': 'hsa-miR-2'}\n"
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "pipelines"
        / "standardized_predictors"
        / "miraw"
        / "miraw_preprocessing.sh"
    )
    subprocess.run(
        ["bash", str(script), str(raw), str(out)],
        check=True,
        env={**os.environ, "MIRAW_THREADS": "2", "MIRAW_SORT_MEMORY": "32M"},
    )

    df = pd.read_csv(out, sep="\t", keep_default_na=False)
    assert df.to_dict("records") == [
        {
            "Ensembl_ID": "ENSG000001",
            "Raw_Gene_Name": "GENE1",
            "miRNA_Name": "hsa-miR-1",
            "Score": 0.9,
        },
        {
            "Ensembl_ID": "ENSG000002",
            "Raw_Gene_Name": "",
            "miRNA_Name": "hsa-miR-2",
            "Score": 0.4,
        },
    ]


def test_load_miraw_predictions_reads_preprocessed_tsv(tmp_path):
    preprocessed = tmp_path / "best_per_pair.tsv.gz"
    with gzip.open(preprocessed, "wt", encoding="utf-8") as handle:
        handle.write("Ensembl_ID\tRaw_Gene_Name\tmiRNA_Name\tScore\n")
        handle.write("ENSG000001.5\tGENE1\thsa-miR-1\t0.9\n")
        handle.write("ENSG000001\tGENE1\thsa-miR-1\t0.9\n")

    df = load_miraw_predictions(preprocessed)

    assert df.to_dict("records") == [
        {
            "Ensembl_ID": "ENSG000001",
            "Raw_Gene_Name": "GENE1",
            "miRNA_Name": "hsa-miR-1",
            "Score": 0.9,
        }
    ]


def test_create_gene_name_mapping_from_ensembl_v115_gtf(tmp_path):
    gtf = tmp_path / "Homo_sapiens.GRCh38.115.gtf.gz"
    with gzip.open(gtf, "wt", encoding="utf-8") as handle:
        handle.write("# header\n")
        handle.write(
            '1\tensembl_havana\tgene\t1\t10\t.\t+\t.\tgene_id "ENSG000001.5"; '
            'gene_version "5"; gene_name "GENE1"; gene_source "ensembl_havana"; '
            'gene_biotype "protein_coding";\n'
        )
        handle.write(
            '1\tensembl_havana\ttranscript\t1\t10\t.\t+\t.\tgene_id "ENSG000001.5"; '
            'transcript_id "ENST000001.1"; gene_name "SHOULD_NOT_OVERRIDE";\n'
        )
        handle.write(
            '1\tensembl_havana\tgene\t20\t30\t.\t+\t.\tgene_id "ENSG000002"; '
            'gene_name "GENE2";\n'
        )

    assert create_ensembl_to_gene_name_mapping_from_gtf(gtf) == {
        "ENSG000001": "GENE1",
        "ENSG000002": "GENE2",
    }


def test_miraw_annotation_keeps_rows_without_gtf_gene_name(tmp_path):
    raw = tmp_path / "best_per_pair.txt"
    raw.write_text(
        "{'GeneName': 'ENSG000001.5__OLD1', 'miRNA': 'hsa-miR-1', 'Prediction': 0.2}\n"
        "{'GeneName': 'ENSG000002__biotype=protein_coding', 'miRNA': 'hsa-miR-1', 'Prediction': 0.8}\n",
        encoding="utf-8",
    )
    df = load_miraw_predictions(raw)

    annotated = map_ensembl_to_gene_name(
        df,
        {"ENSG000001": "GENE1"},
        ensembl_id_column="Ensembl_ID",
        gene_name_column="Gene_Name",
        raw_gene_name_column="Raw_Gene_Name",
    )

    assert annotated["Ensembl_ID"].tolist() == ["ENSG000001", "ENSG000002"]
    assert annotated["Gene_Name"].tolist() == ["GENE1", "ENSG000002"]


def test_miraw_final_output_is_one_clean_row_per_gene_mirna_pair(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text(
        "{'GeneName': 'ENSG000001.1__GENE1', 'miRNA': 'hsa-miR-1', 'Prediction': 0.2}\n"
        "{'GeneName': 'ENSG000001.1__GENE1', 'miRNA': 'hsa-miR-1', 'Prediction': 0.9}\n",
        encoding="utf-8",
    )
    df = load_miraw_predictions(raw)
    collapsed = collapse_to_best_score_per_pair(
        df,
        ensembl_id_column="Ensembl_ID",
        mirna_name_column="miRNA_Name",
        score_column="Score",
    )
    collapsed["miRNA_ID"] = "MIMAT0000001"
    annotated = map_ensembl_to_gene_name(
        collapsed,
        {},
        ensembl_id_column="Ensembl_ID",
        gene_name_column="Gene_Name",
        raw_gene_name_column="Raw_Gene_Name",
    )
    final = build_output_table(
        annotated,
        final_columns=["Ensembl_ID", "Gene_Name", "miRNA_ID", "miRNA_Name", "Score"],
        ensembl_id_column="Ensembl_ID",
        mimat_column="miRNA_ID",
        score_column="Score",
    )

    assert final.to_dict("records") == [
        {
            "Ensembl_ID": "ENSG000001",
            "Gene_Name": "GENE1",
            "miRNA_ID": "MIMAT0000001",
            "miRNA_Name": "hsa-miR-1",
            "Score": 0.9,
        }
    ]
    assert not final.isna().any().any()
