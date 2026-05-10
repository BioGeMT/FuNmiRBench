import pandas as pd

import funmirbench.comparison_plots as comparison_plots


def test_common_plots_exclude_ultra_sparse_predictors(tmp_path, monkeypatch):
    joined = pd.DataFrame(
        {
            "dataset_id": ["D001"] * 5,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"],
            "logFC": [2.0, 1.5, 0.3, -0.2, -1.4],
            "FDR": [0.01, 0.02, 0.4, 0.8, 0.03],
            "score_a": [0.9, 0.7, 0.4, 0.2, 0.1],
            "score_b": [0.8, 0.6, 0.5, 0.3, 0.2],
            "score_sparse": [0.95, None, None, None, None],
        }
    )
    evaluation = {"plots": {}}
    metric_rows = [
        {"tool_id": "a", "coverage": 1.0},
        {"tool_id": "b", "coverage": 1.0},
        {"tool_id": "sparse", "coverage": 0.2},
    ]
    captured = {}

    def capture_pr(comparisons, *, dataset_id, out_path):
        captured["pr"] = comparisons
        out_path.write_text("pr", encoding="utf-8")

    def capture_roc(comparisons, *, dataset_id, out_path):
        captured["roc"] = comparisons
        out_path.write_text("roc", encoding="utf-8")

    def capture_gsea(comparisons, *, dataset_id, out_path):
        captured["gsea"] = comparisons
        out_path.write_text("gsea", encoding="utf-8")

    monkeypatch.setattr(comparison_plots, "_plot_predictor_pr_curves", capture_pr)
    monkeypatch.setattr(comparison_plots, "_plot_predictor_roc_curves", capture_roc)
    monkeypatch.setattr(comparison_plots, "_plot_predictor_gsea_curves", capture_gsea)

    written = comparison_plots.write_common_comparison_plots(
        joined,
        evaluation=evaluation,
        dataset_metric_rows=metric_rows,
        plots_dir=tmp_path / "plots",
        dataset_id="D001",
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        min_common_coverage=0.5,
    )

    assert len(written) == 3
    assert set(captured) == {"pr", "roc", "gsea"}
    for key in captured:
        assert [item["tool_id"] for item in captured[key]] == ["a", "b"]
        assert all(len(item["y_true"]) == 5 for item in captured[key])
    assert evaluation["plots"]["predictor_pr_curves"].endswith("precision_recall_common.png")
    assert evaluation["plots"]["predictor_roc_curves"].endswith("roc_common.png")
    assert evaluation["plots"]["predictor_gsea_curves"].endswith("gsea_common.png")
