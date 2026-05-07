import csv

import pytest

from pipelines.standardized_predictors.targetscan.utils import (
    MirnaEntry,
    step_write_standardized_predictions,
)


def test_targetscan_surviving_duplicate_gene_mirna_pairs_raise(tmp_path):
    summary_counts = tmp_path / "Summary_Counts.all_predictions.txt"
    with summary_counts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Transcript ID",
                "Gene Symbol",
                "miRNA family",
                "Species ID",
                "Total num conserved sites",
                "Number of conserved 8mer sites",
                "Number of conserved 7mer-m8 sites",
                "Number of conserved 7mer-1a sites",
                "Total num nonconserved sites",
                "Number of nonconserved 8mer sites",
                "Number of nonconserved 7mer-m8 sites",
                "Number of nonconserved 7mer-1a sites",
                "Number of 6mer sites",
                "Representative miRNA",
                "Total context++ score",
                "Cumulative weighted context++ score",
                "Aggregate PCT",
                "Predicted occupancy - low miRNA",
                "Predicted occupancy - high miRNA",
                "Predicted occupancy - transfected miRNA",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for transcript_id, score in (("ENST000001.1", "-0.2"), ("ENST000002.1", "-0.4")):
            writer.writerow(
                {
                    "Transcript ID": transcript_id,
                    "Gene Symbol": "GENE1",
                    "miRNA family": "seed",
                    "Species ID": "9606",
                    "Total num conserved sites": "0",
                    "Number of conserved 8mer sites": "0",
                    "Number of conserved 7mer-m8 sites": "0",
                    "Number of conserved 7mer-1a sites": "0",
                    "Total num nonconserved sites": "1",
                    "Number of nonconserved 8mer sites": "0",
                    "Number of nonconserved 7mer-m8 sites": "1",
                    "Number of nonconserved 7mer-1a sites": "0",
                    "Number of 6mer sites": "0",
                    "Representative miRNA": "hsa-miR-1-5p",
                    "Total context++ score": score,
                    "Cumulative weighted context++ score": score,
                    "Aggregate PCT": "0.1",
                    "Predicted occupancy - low miRNA": "NULL",
                    "Predicted occupancy - high miRNA": "NULL",
                    "Predicted occupancy - transfected miRNA": "NULL",
                }
            )

    tx_index = {
        "rep_txs_by_gene_id": {
            "ENSG000001": [
                ("ENST000001.1", 10, "GENE1"),
                ("ENST000002.1", 8, "GENE1"),
            ]
        },
        "targetscan_gene_id_by_tx": {
            "ENST000001.1": "ENSG000001",
            "ENST000002.1": "ENSG000001",
        },
    }
    ensembl_tables = {
        "tx_to_gene": {
            "ENST000001": "ENSG000001",
            "ENST000002": "ENSG000001",
        },
        "gene_to_name": {"ENSG000001": "GENE1"},
    }
    mirna_annotations = {
        "hsa-miR-1-5p": MirnaEntry(
            mirna_id="MIMAT0000001",
            mirna_name="hsa-miR-1-5p",
        )
    }

    with pytest.raises(ValueError, match="Unexpected duplicate gene-miRNA rows remain"):
        step_write_standardized_predictions(
            summary_counts,
            tx_index=tx_index,
            ensembl_tables=ensembl_tables,
            mirna_annotations=mirna_annotations,
            out_predictions_dir=tmp_path / "predictions",
        )
