"""Tests for funmirbench.evaluate."""

import pandas as pd
import pytest

import funmirbench.evaluate as evaluate_module
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
                "positive_coverage": 0.5,
                "aps": 0.75,
                "pr_auc": 0.70,
                "spearman": 0.40,
                "auroc": 0.80,
            }
        ],
        tmp_path,
    )
    assert set(metric_tables) == {"coverage", "positive_coverage", "aps", "pr_auc", "spearman", "auroc"}
    aps_lines = (tmp_path / "aps_per_experiment.tsv").read_text(encoding="utf-8").splitlines()
    assert len(aps_lines) == 2
    assert "D001" in aps_lines[1]
    assert "NA" in aps_lines[1]
    coverage_lines = (tmp_path / "coverage_per_experiment.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(coverage_lines) == 2
    assert coverage_lines[1].endswith("0.75")
    positive_coverage_lines = (tmp_path / "positive_coverage_per_experiment.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(positive_coverage_lines) == 2
    assert positive_coverage_lines[1].endswith("0.5")


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
                "positive_coverage": 0.5,
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
                "positive_coverage": 0.25,
                "aps": 0.5,
                "pr_auc": 0.48,
                "spearman": -0.2,
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
                "positive_coverage": 1.0,
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
    assert (tmp_path / "plots" / "metrics" / "cross_dataset_metric_distributions.png").is_file()
    assert (tmp_path / "plots" / "coverage" / "positive_coverage_vs_performance.png").is_file()
    assert (tmp_path / "plots" / "ranks" / "positive_background_rank_distributions.png").is_file()
    summary_text = (tmp_path / "tables" / "cross_dataset_predictor_summary.tsv").read_text(encoding="utf-8")
    assert "aps_mean" in summary_text
    assert outputs["tables"]["cross_dataset_predictor_summary"].endswith("cross_dataset_predictor_summary.tsv")
    assert outputs["plots"]["positive_coverage_vs_performance"].endswith(
        "positive_coverage_vs_performance.png"
    )
    assert "cross_dataset_metric_heatmap" not in outputs["plots"]
    assert "coverage_vs_performance" not in outputs["plots"]


def test_metric_plot_limits_allow_negative_spearman():
    assert evaluate_module._metric_plot_limits("spearman") == (-1.02, 1.02)
    assert evaluate_module._metric_plot_limits("aps") == (0.0, 1.02)


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


def test_evaluate_uses_perturbation_aware_gt_labels(tmp_path):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001", "D001"],
            "mirna": ["hsa-miR-demo", "hsa-miR-demo"],
            "perturbation": ["OE", "OE"],
            "gene_id": ["ENSG1", "ENSG2"],
            "logFC": [-2.0, 2.0],
            "FDR": [0.01, 0.01],
            "score_mock": [0.9, 0.1],
        }
    )
    result = evaluate_joined_dataframe(
        joined,
        plots_dir=tmp_path / "plots",
        reports_dir=tmp_path / "reports",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        predictor_top_fraction=0.10,
        perturbation="OE",
    )
    assert result["metric_rows"][0]["aps"] == pytest.approx(1.0)
    assert result["metric_rows"][0]["auroc"] == pytest.approx(1.0)


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
    assert "mock_pr_curve" in result["plots"]
    assert "mock_roc_curve" in result["plots"]
    assert "cheating_pr_curve" in result["plots"]
    assert "predictor_pr_curves" in result["plots"]
    assert "predictor_pr_curves_all_scored" in result["plots"]
    assert "predictor_roc_curves" in result["plots"]
    assert "predictor_gsea_curves" in result["plots"]
    assert "top_10pct_positive_heatmap" in result["plots"]
    assert (tmp_path / "plots" / "predictors" / "mock" / "score_vs_expected_effect.png").is_file()
    assert (tmp_path / "plots" / "predictors" / "mock" / "gsea_enrichment.png").is_file()
    assert (tmp_path / "plots" / "predictors" / "mock" / "precision_recall_curve.png").is_file()
    assert (tmp_path / "plots" / "predictors" / "mock" / "roc_curve.png").is_file()
    assert (tmp_path / "plots" / "predictors" / "cheating" / "precision_recall_curve.png").is_file()
    assert (tmp_path / "plots" / "comparisons" / "precision_recall_common.png").is_file()
    assert (tmp_path / "plots" / "comparisons" / "precision_recall_all_scored.png").is_file()
    assert (tmp_path / "plots" / "comparisons" / "roc_common.png").is_file()
    assert (tmp_path / "plots" / "comparisons" / "gsea_common.png").is_file()
    assert (tmp_path / "plots" / "heatmaps" / "top_10pct_positive_genes.png").is_file()


def test_cross_predictor_plots_use_common_scored_rows(tmp_path, monkeypatch):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001"] * 4,
            "mirna": ["hsa-miR-demo"] * 4,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "logFC": [2.0, 0.4, 0.2, -0.1],
            "FDR": [0.01, 0.01, 0.4, 0.8],
            "score_mock": [0.9, 0.8, None, 0.2],
            "score_cheating": [0.95, 0.7, 0.1, None],
        }
    )
    captured = {}

    monkeypatch.setattr(evaluate_module, "_plot_scatter_with_correlation", lambda *args, **kwargs: (0.0, 0.0))
    monkeypatch.setattr(evaluate_module, "_plot_gsea_enrichment", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(evaluate_module, "_plot_single_predictor_pr_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_single_predictor_roc_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_algorithms_vs_genes_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_top_positive_heatmap", lambda *args, **kwargs: False)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_correlation_heatmap", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(evaluate_module, "_write_tool_report", lambda *args, **kwargs: None)

    def capture_pr(comparisons, *, dataset_id, out_path):
        captured["pr"] = comparisons

    def capture_roc(comparisons, *, dataset_id, out_path):
        captured["roc"] = comparisons

    def capture_gsea(comparisons, *, dataset_id, out_path):
        captured["gsea"] = comparisons

    def capture_pr_all(comparisons, *, dataset_id, out_path):
        captured["pr_all"] = comparisons

    monkeypatch.setattr(evaluate_module, "_plot_predictor_pr_curves", capture_pr)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_pr_curves_own_scored", capture_pr_all)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_roc_curves", capture_roc)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_gsea_curves", capture_gsea)

    evaluate_joined_dataframe(
        joined,
        plots_dir=tmp_path / "plots",
        reports_dir=tmp_path / "reports",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        predictor_top_fraction=0.10,
    )

    assert set(captured) == {"pr", "pr_all", "roc", "gsea"}
    for key in ("pr", "roc", "gsea"):
        comparisons = captured[key]
        assert len(comparisons) == 2
        assert all(len(item["y_true"]) == 2 for item in comparisons)
        assert all(int(item["y_true"].sum()) == 1 for item in comparisons)
        assert comparisons[0]["y_true"].tolist() == comparisons[1]["y_true"].tolist()
        assert [item["tool_id"] for item in comparisons] == ["cheating", "mock"]
    own_comparisons = captured["pr_all"]
    assert len(own_comparisons) == 2
    assert sorted((item["tool_id"], len(item["y_true"])) for item in own_comparisons) == [
        ("cheating", 3),
        ("mock", 3),
    ]


def test_top_fraction_mask_uses_exact_top_k_with_deterministic_ties():
    series = pd.Series([1.0, 1.0, 1.0, 0.4, 0.1], index=["ENSG3", "ENSG1", "ENSG2", "ENSG4", "ENSG5"])
    tie_breaker = pd.Series(series.index, index=series.index)

    selected = evaluate_module._top_fraction_mask(series, 0.40, tie_breaker=tie_breaker)

    assert int(selected.sum()) == 2
    assert selected.to_dict() == {
        "ENSG3": False,
        "ENSG1": True,
        "ENSG2": True,
        "ENSG4": False,
        "ENSG5": False,
    }


def test_per_dataset_visuals_use_local_ranks_not_global_ranks(tmp_path, monkeypatch):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001"] * 4,
            "mirna": ["hsa-miR-demo"] * 4,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "logFC": [2.0, 1.5, 0.2, -0.1],
            "FDR": [0.01, 0.02, 0.3, 0.8],
            "score_mock": [0.9, 0.2, 0.5, 0.1],
            "score_cheating": [0.95, 0.88, 0.1, 0.05],
            "global_rank_mock": [0.0, 0.0, 1.0, 1.0],
            "global_rank_cheating": [1.0, 1.0, 0.0, 0.0],
        }
    )
    captured = {}

    monkeypatch.setattr(evaluate_module, "_plot_scatter_with_correlation", lambda *args, **kwargs: (0.0, 0.0))
    monkeypatch.setattr(evaluate_module, "_plot_gsea_enrichment", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(evaluate_module, "_plot_single_predictor_pr_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_single_predictor_roc_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_pr_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_pr_curves_own_scored", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_roc_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_gsea_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_module, "_write_tool_report", lambda *args, **kwargs: None)

    def capture_alg(joined_frame, *, score_cols, rank_cols, tool_ids, dataset_id, out_path, **kwargs):
        captured["alg"] = joined_frame[rank_cols].copy()
        captured["alg_cols"] = rank_cols

    def capture_top(joined_frame, *, rank_cols, tool_ids, dataset_id, out_path, **kwargs):
        captured["top"] = joined_frame[rank_cols].copy()
        captured["top_cols"] = rank_cols
        return False

    def capture_corr(joined_frame, *, rank_cols, tool_ids, dataset_id, out_path, top_fraction):
        captured["corr"] = joined_frame[rank_cols].copy()
        captured["corr_cols"] = rank_cols
        return pd.DataFrame(index=tool_ids, columns=tool_ids, dtype=float)

    monkeypatch.setattr(evaluate_module, "_plot_algorithms_vs_genes_heatmap", capture_alg)
    monkeypatch.setattr(evaluate_module, "_plot_top_positive_heatmap", capture_top)
    monkeypatch.setattr(evaluate_module, "_plot_predictor_correlation_heatmap", capture_corr)

    evaluate_joined_dataframe(
        joined,
        plots_dir=tmp_path / "plots",
        reports_dir=tmp_path / "reports",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        predictor_top_fraction=0.10,
    )

    expected_mock = evaluate_module._rank_scale_scores(joined["score_mock"])
    expected_cheating = evaluate_module._rank_scale_scores(joined["score_cheating"])
    for key in ("alg", "top", "corr"):
        assert captured[f"{key}_cols"] == ["local_rank_cheating", "local_rank_mock"]
        assert captured[key]["local_rank_mock"].tolist() == expected_mock.tolist()
        assert captured[key]["local_rank_cheating"].tolist() == expected_cheating.tolist()


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
    assert metric_row["positive_coverage"] == pytest.approx(2.0 / 3.0)

    report_md = tmp_path / "reports" / "D001__sparse_evaluation_report.md"
    report_pdf = tmp_path / "reports" / "D001__sparse_evaluation_report.pdf"
    report_text = report_md.read_text(encoding="utf-8")
    assert report_pdf.is_file()
    assert "## Snapshot" in report_text
    assert "## Evaluation Rule" in report_text
    assert "rows_total: `5`" in report_text
    assert "rows_scored: `3`" in report_text
    assert "rows_missing_score: `2`" in report_text
    assert "coverage: `0.600000`" in report_text
    assert "positives_total: `3`" in report_text
    assert "positives_scored: `2`" in report_text
    assert "positive_coverage: `0.666667`" in report_text
    assert "precision_recall_curve:" in report_text
    assert "roc_curve:" in report_text
    assert "gsea_enrichment:" in report_text


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
    assert any("Tool: mock | coverage=" in message and "positive_cov=" in message for message in messages)
    assert any("wrote PR/ROC/GSEA comparison plots" in message for message in messages)
    assert any("Evaluation complete: D001" in message for message in messages)
