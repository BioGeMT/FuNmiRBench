"""Tests for built-in predictor generators."""

import pandas as pd

from funmirbench.build_predictions import build_mock_scores
from funmirbench.build_cheating_predictions import build_cheating_scores


def test_build_mock_scores_are_deterministic_and_do_not_follow_de_signal(tmp_path):
    de_path = tmp_path / "de.tsv"
    de_path.write_text(
        "\n".join(
            [
                "gene_id\tlogFC\tFDR",
                "ENSG00000000001\t-3.0\t0.001",
                "ENSG00000000002\t0.1\t0.50",
                "ENSG00000000003\t2.8\t0.002",
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
                "experiment_type": "OE",
            }
        ]
    ).to_csv(experiments_tsv, sep="\t", index=False)

    scores_a = build_mock_scores(experiments_tsv, tmp_path, max_genes_per_mirna=10)
    scores_b = build_mock_scores(experiments_tsv, tmp_path, max_genes_per_mirna=10)

    assert scores_a == scores_b
    assert set(scores_a) == {
        ("hsa-miR-test", "ENSG00000000001"),
        ("hsa-miR-test", "ENSG00000000002"),
        ("hsa-miR-test", "ENSG00000000003"),
    }
    assert 0.0 <= scores_a[("hsa-miR-test", "ENSG00000000001")] <= 1.0
    assert 0.0 <= scores_a[("hsa-miR-test", "ENSG00000000003")] <= 1.0
    assert scores_a[("hsa-miR-test", "ENSG00000000001")] != scores_a[("hsa-miR-test", "ENSG00000000003")]


def test_build_cheating_scores_use_perturbation_direction(tmp_path):
    de_path = tmp_path / "de.tsv"
    de_path.write_text(
        "\n".join(
            [
                "gene_id\tlogFC\tFDR",
                "ENSG00000000001\t-2.0\t0.001",
                "ENSG00000000002\t0.2\t0.50",
                "ENSG00000000003\t2.0\t0.001",
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
                "experiment_type": "OE",
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
