"""End-to-end test via a small real-data benchmark config."""

import json
import pathlib
import re
import subprocess
import sys

import pandas as pd
import pytest

from funmirbench import DatasetMeta
from funmirbench import benchmark
from funmirbench.build_cheating_predictions import build_cheating_scores
from funmirbench.build_predictions import build_random_scores, write_tsv


PDF_MEDIA_BOX_PATTERN = re.compile(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]")


def _pdf_media_boxes(pdf_path):
    return [
        (float(width), float(height))
        for width, height in PDF_MEDIA_BOX_PATTERN.findall(pdf_path.read_bytes())
    ]


def test_selected_experiment_paths_applies_filters(tmp_path):
    experiments_tsv = tmp_path / "experiments.tsv"
    pd.DataFrame(
        [
            {"id": "A", "de_table_path": "data/experiments/processed/18745741/a.tsv", "mirna_name": "m1"},
            {"id": "B", "de_table_path": "data/experiments/processed/18745741/b.tsv", "mirna_name": "m2"},
        ]
    ).to_csv(experiments_tsv, sep="\t", index=False)

    paths = benchmark.selected_experiment_paths(experiments_tsv, {"id": ["B"]})

    assert paths == ["data/experiments/processed/18745741/b.tsv"]


def test_validate_threshold_sensitive_predictors_requires_matching_metadata(tmp_path):
    cheating_path = tmp_path / "cheating_standardized.tsv"
    cheating_path.write_text("", encoding="utf-8")
    predictions = {"cheating": {"predictor_output_path": str(cheating_path)}}

    with pytest.raises(ValueError, match="no sidecar metadata file"):
        benchmark.validate_threshold_sensitive_predictors(
            predictions,
            root=tmp_path,
            fdr_threshold=0.10,
            abs_logfc_threshold=1.0,
        )

    metadata_path = pathlib.Path(str(cheating_path) + ".meta.json")
    metadata_path.write_text(
        json.dumps({"fdr_threshold": 0.10, "abs_logfc_threshold": 1.0}),
        encoding="utf-8",
    )

    benchmark.validate_threshold_sensitive_predictors(
        predictions,
        root=tmp_path,
        fdr_threshold=0.10,
        abs_logfc_threshold=1.0,
    )

    metadata_path.write_text(
        json.dumps({"fdr_threshold": 0.05, "abs_logfc_threshold": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="was built with thresholds"):
        benchmark.validate_threshold_sensitive_predictors(
            predictions,
            root=tmp_path,
            fdr_threshold=0.10,
            abs_logfc_threshold=1.0,
        )


def test_build_run_dir_name_summarizes_selection(tmp_path):
    experiments = [
        DatasetMeta(
            id="GSE109725_OE_miR_204_5p",
            miRNA="hsa-miR-204-5p",
            cell_line="4T1",
            tissue="breast",
            perturbation="OE",
            organism="Homo sapiens",
            geo_accession="GSE109725",
            data_path="a.tsv",
            root=tmp_path,
        ),
        DatasetMeta(
            id="GSE118315_KO_miR_124_3p",
            miRNA="hsa-miR-124-3p",
            cell_line="iNGN",
            tissue="neuron",
            perturbation="KO",
            organism="Homo sapiens",
            geo_accession="GSE118315",
            data_path="b.tsv",
            root=tmp_path,
        ),
        DatasetMeta(
            id="GSE210778_OE_miR_375_3p",
            miRNA="hsa-miR-375-3p",
            cell_line="HUV-EC-C",
            tissue="endothelium",
            perturbation="OE",
            organism="Homo sapiens",
            geo_accession="GSE210778",
            data_path="c.tsv",
            root=tmp_path,
        ),
    ]

    name = benchmark.build_run_dir_name(
        experiments=experiments,
        tool_ids=["random", "cheating", "targetscan", "mirdb_mirtarget"],
        eval_cfg={
            "fdr_threshold": 0.05,
            "abs_logfc_threshold": 1.0,
            "predictor_top_fraction": 0.10,
        },
        tags=["demo"],
    )

    assert name.startswith(
        "tag-demo__datasets-gse109725-oe-mir-204-5p-gse118315-ko-mir-124-3p-plus1"
    )
    assert "__mirnas-hsa-mir-204-5p-hsa-mir-124-3p-plus1__" in name
    assert "__tools-random-cheating-targetscan-plus1__" in name
    assert "__pert-ko-oe__cell3__fdr0p05-effect1-top10pct" in name

def test_example_end_to_end(tmp_path):
    """Run a small two-predictor benchmark config and check outputs."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tmp_dir = tmp_path
    config = tmp_dir / "benchmark.yaml"
    out_root = tmp_dir / "results"

    experiments = pd.read_csv(repo_root / "metadata" / "mirna_experiment_info.tsv", sep="\t")
    experiments = experiments[
        experiments["id"].isin([
            "GSE109725_OE_miR_204_5p",
            "GSE118315_KO_miR_124_3p",
            "GSE210778_OE_miR_375_3p",
        ])
    ].copy()
    experiments["de_table_path"] = experiments["de_table_path"].map(
        lambda value: str((repo_root / value).resolve())
    )
    experiments_tsv = tmp_dir / "experiments.tsv"
    experiments.to_csv(experiments_tsv, sep="\t", index=False)

    predictions = pd.read_csv(repo_root / "metadata" / "predictions_info.tsv", sep="\t")
    predictions = predictions[predictions["tool_id"].isin(["random", "cheating"])].copy()

    random_path = tmp_dir / "random_standardized.tsv"
    cheating_path = tmp_dir / "cheating_standardized.tsv"
    write_tsv(build_random_scores(experiments_tsv, tmp_dir), random_path)
    write_tsv(
        build_cheating_scores(
            experiments_tsv,
            tmp_dir,
            dataset_ids=[
                "GSE109725_OE_miR_204_5p",
                "GSE118315_KO_miR_124_3p",
                "GSE210778_OE_miR_375_3p",
            ],
        ),
        cheating_path,
    )

    predictions["predictor_output_path"] = predictions["tool_id"].map(
        {
            "random": str(random_path.resolve()),
            "cheating": str(cheating_path.resolve()),
        }
    )
    predictions_tsv = tmp_dir / "predictions.tsv"
    predictions.to_csv(predictions_tsv, sep="\t", index=False)

    config.write_text(
        "\n".join(
            [
                f"experiments_tsv: {experiments_tsv}",
                f"predictions_tsv: {predictions_tsv}",
                "",
                "experiments:",
                "  id: [GSE109725_OE_miR_204_5p, GSE118315_KO_miR_124_3p, GSE210778_OE_miR_375_3p]",
                "",
                "predictors:",
                "  tool_id: [random, cheating]",
                "",
                "evaluation:",
                "  fdr_threshold: 0.05",
                "  abs_logfc_threshold: 1.0",
                "  predictor_top_fraction: 0.10",
                "",
                "tags: [demo, end_to_end]",
                "",
                f"out_dir: {out_root}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    stale_run_dir = out_root / "legacy_run"
    stale_plot = stale_run_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "random_score_vs_logFC.png"
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_text("stale", encoding="utf-8")
    stale_report = stale_run_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "reports" / "GSE109725_OE_miR_204_5p__random_evaluation_report.md"
    stale_report.parent.mkdir(parents=True, exist_ok=True)
    stale_report.write_text("stale", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "funmirbench.benchmark", "--config", str(config)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    summary_paths = list(out_root.glob("*/summary.json"))
    assert len(summary_paths) == 1
    out_dir = summary_paths[0].parent

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "REPORT.pdf").is_file()
    assert (out_dir / "tables" / "per_experiment" / "coverage_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "positive_coverage_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "aps_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "pr_auc_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "combined" / "cross_dataset_predictor_summary.tsv").is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "reports" / "GSE109725_OE_miR_204_5p__random_evaluation_report.md"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "reports" / "GSE109725_OE_miR_204_5p__random_evaluation_report.pdf"
    ).is_file()
    report_pdf = out_dir / "REPORT.pdf"
    dataset_pdf = (
        out_dir
        / "datasets"
        / "GSE109725_OE_miR_204_5p"
        / "reports"
        / "GSE109725_OE_miR_204_5p__random_evaluation_report.pdf"
    )

    joined_files = sorted((out_dir / "datasets").glob("*/joined.tsv"))
    assert [path.parent.name for path in joined_files] == [
        "GSE109725_OE_miR_204_5p",
        "GSE118315_KO_miR_124_3p",
        "GSE210778_OE_miR_375_3p",
    ]

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["out_root"] == str(out_root)
    assert summary["out_dir"] == str(out_dir)
    assert summary["run_dir_name"] == out_dir.name
    assert summary["tags"] == ["demo", "end_to_end"]
    assert summary["readme"].endswith("README.md")
    assert summary["report_pdf"].endswith("REPORT.pdf")
    assert "cross_dataset_outputs" in summary
    assert summary["cross_dataset_outputs"]["tables"]["cross_dataset_predictor_summary"].endswith(
        "cross_dataset_predictor_summary.tsv"
    )
    assert summary["cross_dataset_outputs"]["plots"]["positive_coverage_vs_performance"].endswith(
        "positive_coverage_vs_performance.png"
    )
    assert "coverage_vs_performance" not in summary["cross_dataset_outputs"]["plots"]
    assert "cross_dataset_metric_heatmap" not in summary["cross_dataset_outputs"]["plots"]
    assert set(summary["dataset_ids"]) == {
        "GSE109725_OE_miR_204_5p",
        "GSE210778_OE_miR_375_3p",
        "GSE118315_KO_miR_124_3p",
    }
    assert summary["tool_ids"] == ["random", "cheating"]

    run_media_boxes = _pdf_media_boxes(report_pdf)
    assert len(run_media_boxes) > 1
    assert len(set(run_media_boxes)) == 1
    run_width, run_height = run_media_boxes[0]
    assert run_width == pytest.approx(72.0 * benchmark.REPORT_PAGE_SIZE[0], abs=0.02)
    assert run_height == pytest.approx(72.0 * benchmark.REPORT_PAGE_SIZE[1], abs=0.02)

    dataset_media_boxes = _pdf_media_boxes(dataset_pdf)
    assert len(dataset_media_boxes) == 2
    assert len(set(dataset_media_boxes)) == 1
    dataset_width, dataset_height = dataset_media_boxes[0]
    assert dataset_width == pytest.approx(72.0 * benchmark.REPORT_PAGE_SIZE[0], abs=0.02)
    assert dataset_height == pytest.approx(72.0 * benchmark.REPORT_PAGE_SIZE[1], abs=0.02)

    plots = list(out_dir.rglob("*.png"))
    assert len(plots) == 60
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "predictors" / "random" / "score_vs_expected_effect.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "predictors" / "random" / "gsea_enrichment.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "predictors" / "random" / "precision_recall_curve.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "predictors" / "random" / "roc_curve.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "predictors" / "cheating" / "precision_recall_curve.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "predictors" / "cheating" / "roc_curve.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_coverage_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_positive_coverage_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_aps_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_pr_auc_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_spearman_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_auroc_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "coverage" / "positive_coverage_vs_performance.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_background_local_rank_distributions.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_background_global_rank_distributions.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "heatmaps" / "top_10pct_positive_genes.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "comparisons" / "precision_recall_common.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "comparisons" / "precision_recall_all_scored.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "comparisons" / "roc_common.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "comparisons" / "roc_all_scored.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "comparisons" / "gsea_common.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "GSE109725_OE_miR_204_5p" / "plots" / "comparisons" / "top_100_effect_cdfs.png"
    ).is_file()
    assert stale_plot.exists()
    assert stale_report.exists()

    aps_lines = (out_dir / "tables" / "per_experiment" / "aps_per_experiment.tsv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(aps_lines) == 4
    header = aps_lines[0].split("\t")
    assert "random" in header
    assert "cheating" in header
    coverage_lines = (out_dir / "tables" / "per_experiment" / "coverage_per_experiment.tsv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(coverage_lines) == 4
    positive_coverage_lines = (
        out_dir / "tables" / "per_experiment" / "positive_coverage_per_experiment.tsv"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(positive_coverage_lines) == 4


def test_run_benchmark_syncs_missing_experiment_tables(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    experiments_tsv = tmp_path / "experiments.tsv"
    predictions_tsv = tmp_path / "predictions.tsv"
    results_dir = tmp_path / "results"

    pd.DataFrame(
        [
            {
                "id": "T001",
                "mirna_name": "hsa-miR-test",
                "tested_cell_line": "HeLa",
                "tissue": "cervix",
                "experiment_type": "OE",
                "organism": "Homo sapiens",
                "gse_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE000001",
                "de_table_path": "data/experiments/processed/18745741/demo.tsv",
            }
        ]
    ).to_csv(experiments_tsv, sep="\t", index=False)

    score_path = tmp_path / "scores.tsv"
    score_path.write_text(
        "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
        "ENSG1\tGENE1\t\thsa-miR-test\t0.9\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tool_id": "random",
                "predictor_output_path": str(score_path),
            }
        ]
    ).to_csv(predictions_tsv, sep="\t", index=False)

    config.write_text(
        "\n".join(
            [
                f"experiments_tsv: {experiments_tsv}",
                f"predictions_tsv: {predictions_tsv}",
                "experiments:",
                "  id: [T001]",
                "predictors:",
                "  tool_id: [random]",
                f"out_dir: {results_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sync_calls = []

    def fake_sync_zenodo_experiments(paths, *, repo=None, registry=None, token=None, timeout=120, force=False):
        sync_calls.append((paths, repo, token, timeout, force))
        dest = repo / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            "gene_id\tlogFC\tFDR\tPValue\n"
            "ENSG1\t-2.0\t0.01\t0.001\n",
            encoding="utf-8",
        )
        return [dest]

    monkeypatch.setattr(benchmark, "sync_zenodo_experiments", fake_sync_zenodo_experiments)
    monkeypatch.setattr(
        benchmark,
        "evaluate_joined_dataframe",
        lambda *args, **kwargs: {
            "metric_rows": [],
            "plots": [],
            "predictor_correlation_tsv": None,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "write_metric_tables",
        lambda metric_rows, tables_dir, logger=None: {
            "aps_per_experiment": str(tables_dir / "aps.tsv")
        },
    )
    monkeypatch.setattr(
        benchmark,
        "write_cross_dataset_summaries",
        lambda metric_rows, tables_dir, plots_dir, **kwargs: {"tables": {}, "plots": {}},
    )

    out_dir = benchmark.run_benchmark(config)

    assert out_dir.parent == results_dir.resolve()
    assert sync_calls == [(["data/experiments/processed/18745741/demo.tsv"], tmp_path, None, 120, False)]
    joined = pd.read_csv(out_dir / "datasets" / "T001" / "joined.tsv", sep="\t")
    assert joined["gene_id"].tolist() == ["ENSG1"]
    assert joined["score_random"].tolist() == [0.9]


def test_run_benchmark_passes_post_evaluation_joined_frames(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    experiments_tsv = tmp_path / "experiments.tsv"
    predictions_tsv = tmp_path / "predictions.tsv"
    results_dir = tmp_path / "results"

    pd.DataFrame(
        [
            {
                "id": "T001",
                "mirna_name": "hsa-miR-test",
                "tested_cell_line": "HeLa",
                "tissue": "cervix",
                "experiment_type": "OE",
                "organism": "Homo sapiens",
                "gse_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE000001",
                "de_table_path": "data/experiments/processed/18745741/demo.tsv",
            }
        ]
    ).to_csv(experiments_tsv, sep="\t", index=False)

    score_path = tmp_path / "scores.tsv"
    score_path.write_text(
        "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
        "ENSG1\tGENE1\t\thsa-miR-test\t0.9\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tool_id": "random",
                "predictor_output_path": str(score_path),
            }
        ]
    ).to_csv(predictions_tsv, sep="\t", index=False)

    config.write_text(
        "\n".join(
            [
                f"experiments_tsv: {experiments_tsv}",
                f"predictions_tsv: {predictions_tsv}",
                "experiments:",
                "  id: [T001]",
                "predictors:",
                "  tool_id: [random]",
                f"out_dir: {results_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_sync_zenodo_experiments(paths, *, repo=None, registry=None, token=None, timeout=120, force=False):
        dest = repo / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            "gene_id\tlogFC\tFDR\tPValue\n"
            "ENSG1\t-2.0\t0.01\t0.001\n",
            encoding="utf-8",
        )
        return [dest]

    captured = {}

    def fake_evaluate_joined_dataframe(joined, *args, **kwargs):
        joined["local_rank_random"] = [0.8]
        return {
            "metric_rows": [],
            "plots": [],
            "predictor_correlation_tsv": None,
        }

    def fake_write_cross_dataset_summaries(metric_rows, tables_dir, plots_dir, **kwargs):
        captured["joined_frames"] = kwargs.get("joined_frames")
        return {"tables": {}, "plots": {}}

    monkeypatch.setattr(benchmark, "sync_zenodo_experiments", fake_sync_zenodo_experiments)
    monkeypatch.setattr(benchmark, "evaluate_joined_dataframe", fake_evaluate_joined_dataframe)
    monkeypatch.setattr(
        benchmark,
        "write_metric_tables",
        lambda metric_rows, tables_dir, logger=None: {"aps_per_experiment": str(tables_dir / "aps.tsv")},
    )
    monkeypatch.setattr(benchmark, "write_cross_dataset_summaries", fake_write_cross_dataset_summaries)

    benchmark.run_benchmark(config)

    assert len(captured["joined_frames"]) == 1
    assert "local_rank_random" in captured["joined_frames"][0].columns
