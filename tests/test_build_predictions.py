"""Tests for built-in predictor generators."""

import pandas as pd

from funmirbench.build_cheating_predictions import build_cheating_scores


def test_build_cheating_scores_separates_positive_and_negative(tmp_path):
    de_path = tmp_path / "de.tsv"
    de_path.write_text(
        "\n".join(
            [
                "gene_id\tlogFC\tFDR",
                "ENSG00000000001\t2.0\t0.001",
                "ENSG00000000002\t0.2\t0.50",
                "ENSG00000000003\t-0.3\t0.20",
                "",
            ]
        ),
        encoding="utf-8",
    )

    experiments_tsv = tmp_path / "experiments.tsv"
    pd.DataFrame(
        [
            {
                "id": "D001",
                "mirna_name": "hsa-miR-test",
                "de_table_path": "de.tsv",
            }
        ]
    ).to_csv(experiments_tsv, sep="\t", index=False)

    scores = build_cheating_scores(experiments_tsv, tmp_path, dataset_ids=["D001"])

    positive = scores[("hsa-miR-test", "ENSG00000000001")]
    negative_a = scores[("hsa-miR-test", "ENSG00000000002")]
    negative_b = scores[("hsa-miR-test", "ENSG00000000003")]

    assert 0.55 <= positive <= 0.95
    assert 0.0 <= negative_a <= 0.95
    assert 0.0 <= negative_b <= 0.95
    assert positive > negative_a
    assert positive > negative_b
