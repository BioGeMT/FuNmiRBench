import json
import pathlib

from scripts.audit_report_bundle import audit_run


def _write_minimal_pdf(path: pathlib.Path):
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<< /Type /Page /MediaBox [0 0 595.44 841.68] >>endobj\n"
        b"%%EOF\n"
    )


def test_audit_run_flags_missing_auroc_metric_table(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_minimal_pdf(run_dir / "REPORT.pdf")
    tables_dir = run_dir / "tables" / "combined"
    tables_dir.mkdir(parents=True)
    summary_table = tables_dir / "cross_dataset_predictor_summary.tsv"
    summary_table.write_text(
        "tool_id\tcoverage_mean\taps_mean\n"
        "targetscan\t0.50\t0.12\n",
        encoding="utf-8",
    )
    summary = {
        "out_dir": str(run_dir),
        "report_pdf": str(run_dir / "REPORT.pdf"),
        "metric_tables": {
            "coverage": "coverage.tsv",
            "positive_coverage": "positive_coverage.tsv",
            "aps": "aps.tsv",
            "pr_auc": "pr_auc.tsv",
            "spearman": "spearman.tsv",
        },
        "cross_dataset_outputs": {
            "tables": {"cross_dataset_predictor_summary": str(summary_table)},
            "plots": {
                "cross_dataset_coverage_distribution": "coverage.png",
                "cross_dataset_positive_coverage_distribution": "positive_coverage.png",
                "cross_dataset_aps_distribution": "aps.png",
                "cross_dataset_pr_auc_distribution": "pr_auc.png",
                "cross_dataset_spearman_distribution": "spearman.png",
                "cross_dataset_auroc_distribution": "auroc.png",
                "positive_background_local_rank_distributions": "local.png",
                "positive_background_global_rank_distributions": "global.png",
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    issues = audit_run(run_dir, min_headline_coverage=0.10)

    assert any("metric_tables is missing: auroc" in issue.message for issue in issues)


def test_audit_run_warns_about_sparse_non_oracle_headline_candidates(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_minimal_pdf(run_dir / "REPORT.pdf")
    tables_dir = run_dir / "tables" / "combined"
    tables_dir.mkdir(parents=True)
    summary_table = tables_dir / "cross_dataset_predictor_summary.tsv"
    summary_table.write_text(
        "tool_id\tcoverage_mean\taps_mean\n"
        "tec-mitarget\t0.001\t0.173\n"
        "cheating\t1.0\t0.999\n",
        encoding="utf-8",
    )
    summary = {
        "out_dir": str(run_dir),
        "report_pdf": str(run_dir / "REPORT.pdf"),
        "metric_tables": {
            "coverage": "coverage.tsv",
            "positive_coverage": "positive_coverage.tsv",
            "aps": "aps.tsv",
            "pr_auc": "pr_auc.tsv",
            "spearman": "spearman.tsv",
            "auroc": "auroc.tsv",
        },
        "cross_dataset_outputs": {
            "tables": {"cross_dataset_predictor_summary": str(summary_table)},
            "plots": {
                "cross_dataset_coverage_distribution": "coverage.png",
                "cross_dataset_positive_coverage_distribution": "positive_coverage.png",
                "cross_dataset_aps_distribution": "aps.png",
                "cross_dataset_pr_auc_distribution": "pr_auc.png",
                "cross_dataset_spearman_distribution": "spearman.png",
                "cross_dataset_auroc_distribution": "auroc.png",
                "positive_background_local_rank_distributions": "local.png",
                "positive_background_global_rank_distributions": "global.png",
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    issues = audit_run(run_dir, min_headline_coverage=0.10)

    assert any("Sparse non-oracle predictors" in issue.message for issue in issues)
    assert not any("cheating" in issue.message for issue in issues)
