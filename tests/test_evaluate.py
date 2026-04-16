"""Tests for funmirbench.evaluate."""

import pandas as pd
import pytest

from funmirbench.evaluate import evaluate_joined_dataframe, write_metric_tables


def test_write_metric_tables_keeps_rows_with_missing_ids(tmp_path):
    metric_tables = write_metric_tables(
        [
            {
                "dataset_id": "D001",
                "mirna": "hsa-miR-demo",
                "cell_line": "HeLa",
                "perturbation": "OE",
                "geo_accession": None,
                "tool_id": "mock",
                "coverage": 0.75,
                "aps": 0.75,
                "pr_auc": 0.70,
                "spearman": 0.40,
                "auroc": 0.80,
            }
        ],
        tmp_path,
    )
    assert set(metric_tables) == {"coverage", "aps", "pr_auc", "spearman", "auroc"}
    aps_lines = (tmp_path / "aps_per_experiment.tsv").read_text(encoding="utf-8").splitlines()
    assert len(aps_lines) == 2
    assert "D001" in aps_lines[1]
    assert "NA" in aps_lines[1]
    coverage_lines = (tmp_path / "coverage_per_experiment.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(coverage_lines) == 2
    assert coverage_lines[1].endswith("0.75")


def test_evaluate_rejects_single_class_labels(tmp_path):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001", "D001"],
            "mirna": ["hsa-miR-demo", "hsa-miR-demo"],
            "gene_id": ["ENSG1", "ENSG2"],
            "logFC": [2.0, 3.0],
            "FDR": [0.01, 0.001],
            "score_mock": [0.8, 0.2],
        }
    )
    with pytest.raises(ValueError, match="No negatives remain"):
        evaluate_joined_dataframe(
            joined,
            plots_dir=tmp_path / "plots",
            reports_dir=tmp_path / "reports",
            fdr_threshold=0.05,
            abs_logfc_threshold=1.0,
            predictor_top_fraction=0.10,
        )


def test_evaluate_writes_combined_comparison_plots(tmp_path):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001", "D001", "D001", "D001"],
            "mirna": ["hsa-miR-demo"] * 4,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "logFC": [2.0, 1.5, 0.2, -0.1],
            "FDR": [0.01, 0.02, 0.3, 0.8],
            "score_mock": [0.9, 0.2, 0.5, 0.1],
            "score_cheating": [0.95, 0.88, 0.1, 0.05],
        }
    )
    result = evaluate_joined_dataframe(
        joined,
        plots_dir=tmp_path / "plots",
        reports_dir=tmp_path / "reports",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        predictor_top_fraction=0.10,
    )
    assert "mock_scatter" in result["plots"]
    assert "predictor_pr_curves" in result["plots"]
    assert "predictor_roc_curves" in result["plots"]
    assert "mock_pr_curve" not in result["plots"]
    assert "mock_roc_curve" not in result["plots"]
    assert (tmp_path / "plots" / "D001" / "mock_score_vs_logFC.png").is_file()
    assert (tmp_path / "plots" / "D001" / "predictor_pr_curves.png").is_file()
    assert (tmp_path / "plots" / "D001" / "predictor_roc_curves.png").is_file()
    assert not (tmp_path / "plots" / "D001" / "mock_pr_curve.png").exists()
    assert not (tmp_path / "plots" / "D001" / "mock_roc_curve.png").exists()


def test_evaluate_uses_only_existing_pairs_and_reports_coverage(tmp_path):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001"] * 5,
            "mirna": ["hsa-miR-demo"] * 5,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"],
            "logFC": [2.0, 1.5, 0.2, -0.1, -1.4],
            "FDR": [0.01, 0.02, 0.3, 0.8, 0.03],
            "score_sparse": [0.9, None, 0.5, None, 0.1],
        }
    )
    result = evaluate_joined_dataframe(
        joined,
        plots_dir=tmp_path / "plots",
        reports_dir=tmp_path / "reports",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        predictor_top_fraction=0.10,
    )
    metric_row = result["metric_rows"][0]
    assert metric_row["rows_total"] == 5
    assert metric_row["rows_scored"] == 3
    assert metric_row["rows_missing_score"] == 2
    assert metric_row["coverage"] == pytest.approx(0.6)

    report_text = (
        tmp_path / "reports" / "D001__sparse_evaluation_report.txt"
    ).read_text(encoding="utf-8")
    assert "rows_total: 5" in report_text
    assert "rows_scored: 3" in report_text
    assert "rows_missing_score: 2" in report_text
    assert "coverage: 0.600000" in report_text


def test_evaluate_uses_supplied_logger(tmp_path):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001", "D001", "D001", "D001"],
            "mirna": ["hsa-miR-demo"] * 4,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "logFC": [2.0, 1.5, 0.2, -0.1],
            "FDR": [0.01, 0.02, 0.3, 0.8],
            "score_mock": [0.9, 0.2, 0.5, 0.1],
            "score_cheating": [0.95, 0.88, 0.1, 0.05],
        }
    )
    messages = []
    evaluate_joined_dataframe(
        joined,
        plots_dir=tmp_path / "plots",
        reports_dir=tmp_path / "reports",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        predictor_top_fraction=0.10,
        logger=messages.append,
    )
    assert any("Evaluation start: D001" in message for message in messages)
    assert any("Tool: mock | coverage=" in message for message in messages)
    assert any("wrote PR/ROC comparison plots" in message for message in messages)
    assert any("Evaluation complete: D001" in message for message in messages)
