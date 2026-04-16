"""Tests for funmirbench.evaluate."""

import pandas as pd
import pytest

from funmirbench.evaluate import (
    evaluate_joined_dataframe,
    write_cross_dataset_summaries,
    write_metric_tables,
)


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


def test_write_cross_dataset_summaries_creates_table_and_plots(tmp_path):
    joined_frames = [
        pd.DataFrame(
            {
                "dataset_id": ["D001", "D001", "D001"],
                "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                "logFC": [2.0, 0.1, -1.5],
                "FDR": [0.01, 0.6, 0.02],
                "global_rank_mock": [0.9, 0.3, 0.7],
                "global_rank_cheating": [0.98, 0.2, 0.8],
            }
        ),
        pd.DataFrame(
            {
                "dataset_id": ["D002", "D002", "D002"],
                "gene_id": ["ENSG4", "ENSG5", "ENSG6"],
                "logFC": [1.8, -0.2, -1.7],
                "FDR": [0.03, 0.4, 0.01],
                "global_rank_mock": [0.8, 0.4, 0.6],
                "global_rank_cheating": [0.96, 0.1, 0.75],
            }
        ),
    ]
    outputs = write_cross_dataset_summaries(
        [
            {
                "dataset_id": "D001",
                "mirna": "mir-1",
                "cell_line": "HeLa",
                "perturbation": "OE",
                "geo_accession": "GSE1",
                "tool_id": "mock",
                "coverage": 0.8,
                "aps": 0.7,
                "pr_auc": 0.68,
                "spearman": 0.4,
                "auroc": 0.77,
            },
            {
                "dataset_id": "D002",
                "mirna": "mir-2",
                "cell_line": "A549",
                "perturbation": "KO",
                "geo_accession": "GSE2",
                "tool_id": "mock",
                "coverage": 0.6,
                "aps": 0.5,
                "pr_auc": 0.48,
                "spearman": 0.2,
                "auroc": 0.65,
            },
            {
                "dataset_id": "D001",
                "mirna": "mir-1",
                "cell_line": "HeLa",
                "perturbation": "OE",
                "geo_accession": "GSE1",
                "tool_id": "cheating",
                "coverage": 1.0,
                "aps": 0.95,
                "pr_auc": 0.94,
                "spearman": 0.85,
                "auroc": 0.98,
            },
        ],
        tmp_path / "tables",
        tmp_path / "plots",
        joined_frames=joined_frames,
    )
    assert (tmp_path / "tables" / "cross_dataset_predictor_summary.tsv").is_file()
    assert (tmp_path / "plots" / "cross_dataset_metric_heatmap.png").is_file()
    assert (tmp_path / "plots" / "cross_dataset_metric_distributions.png").is_file()
    assert (tmp_path / "plots" / "coverage_vs_performance.png").is_file()
    assert (tmp_path / "plots" / "positive_background_rank_distributions.png").is_file()
    summary_text = (tmp_path / "tables" / "cross_dataset_predictor_summary.tsv").read_text(encoding="utf-8")
    assert "aps_mean" in summary_text
    assert outputs["tables"]["cross_dataset_predictor_summary"].endswith("cross_dataset_predictor_summary.tsv")
    assert outputs["plots"]["coverage_vs_performance"].endswith("coverage_vs_performance.png")


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
    assert "mock_gsea_enrichment" in result["plots"]
    assert "predictor_pr_curves" in result["plots"]
    assert "predictor_roc_curves" in result["plots"]
    assert "top_10pct_positive_heatmap" in result["plots"]
    assert "mock_pr_curve" not in result["plots"]
    assert "mock_roc_curve" not in result["plots"]
    assert (tmp_path / "plots" / "mock_score_vs_logFC.png").is_file()
    assert (tmp_path / "plots" / "mock_gsea_enrichment.png").is_file()
    assert (tmp_path / "plots" / "predictor_pr_curves.png").is_file()
    assert (tmp_path / "plots" / "predictor_roc_curves.png").is_file()
    assert (tmp_path / "plots" / "top_10pct_positive_heatmap.png").is_file()
    assert not (tmp_path / "plots" / "mock_pr_curve.png").exists()
    assert not (tmp_path / "plots" / "mock_roc_curve.png").exists()


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

    report_md = tmp_path / "reports" / "D001__sparse_evaluation_report.md"
    report_pdf = tmp_path / "reports" / "D001__sparse_evaluation_report.pdf"
    report_text = report_md.read_text(encoding="utf-8")
    assert report_pdf.is_file()
    assert "rows_total: `5`" in report_text
    assert "rows_scored: `3`" in report_text
    assert "rows_missing_score: `2`" in report_text
    assert "coverage: `0.600000`" in report_text


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
