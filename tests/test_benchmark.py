"""End-to-end tests for the benchmark runner."""

import datetime as dt
import json
import pathlib
import re

import pandas as pd
import pytest

from funmirbench import DatasetMeta
from funmirbench import benchmark


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


def test_run_benchmark_stops_on_experiment_validation_errors(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    experiments_tsv = tmp_path / "experiments.tsv"
    predictions_tsv = tmp_path / "predictions.tsv"
    results_dir = tmp_path / "results"

    de_table = tmp_path / "data" / "experiments" / "processed" / "18745741" / "demo.tsv"
    de_table.parent.mkdir(parents=True, exist_ok=True)
    de_table.write_text(
        "gene_id\tlogFC\tFDR\tPValue\n"
        "ENSG1\t-2.0\t0.01\t0.001\n",
        encoding="utf-8",
    )
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
                "tool_id": "tool_a",
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
                "  tool_id: [tool_a]",
                f"out_dir: {results_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(benchmark, "sync_zenodo_experiments", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="Experiment validation failed") as excinfo:
        benchmark.run_benchmark(config)

    message = str(excinfo.value)
    assert "T001 [ground_truth_classes]" in message
    assert "no negative genes" in message
    assert not results_dir.exists()


def test_build_run_dir_name_uses_date_only(tmp_path):
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
        )
    ]

    name = benchmark.build_run_dir_name(
        experiments=experiments,
        tool_ids=["targetscan", "mirdb_mirtarget"],
        eval_cfg={"fdr_threshold": 0.05, "abs_logfc_threshold": 1.0},
        tags=["demo"],
        run_date=dt.date(2026, 5, 10),
    )

    assert name == "20260510"


def test_example_end_to_end(tmp_path, monkeypatch):
    """Run a small two-predictor benchmark config and check outputs."""
    tmp_dir = tmp_path
    config = tmp_dir / "benchmark.yaml"
    out_root = tmp_dir / "results"

    de_dir = tmp_dir / "data" / "experiments" / "processed"
    de_dir.mkdir(parents=True)
    (de_dir / "d001.tsv").write_text(
        "gene_id\tlogFC\tFDR\tPValue\n"
        "ENSG1\t-2.0\t0.01\t0.001\n"
        "ENSG2\t0.2\t0.50\t0.500\n"
        "ENSG3\t1.5\t0.02\t0.002\n",
        encoding="utf-8",
    )
    (de_dir / "d002.tsv").write_text(
        "gene_id\tlogFC\tFDR\tPValue\n"
        "ENSG4\t-1.8\t0.01\t0.001\n"
        "ENSG5\t0.1\t0.40\t0.400\n"
        "ENSG6\t1.4\t0.02\t0.002\n",
        encoding="utf-8",
    )
    experiments = pd.DataFrame(
        [
            {
                "id": "D001",
                "mirna_name": "hsa-miR-test",
                "tested_cell_line": "HeLa",
                "tissue": "cervix",
                "experiment_type": "OE",
                "organism": "Homo sapiens",
                "gse_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE000001",
                "de_table_path": "data/experiments/processed/d001.tsv",
            },
            {
                "id": "D002",
                "mirna_name": "hsa-miR-test",
                "tested_cell_line": "HeLa",
                "tissue": "cervix",
                "experiment_type": "OE",
                "organism": "Homo sapiens",
                "gse_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE000002",
                "de_table_path": "data/experiments/processed/d002.tsv",
            },
        ]
    )
    experiments_tsv = tmp_dir / "experiments.tsv"
    experiments.to_csv(experiments_tsv, sep="\t", index=False)

    targetscan_path = tmp_dir / "targetscan_standardized.tsv"
    mirdb_path = tmp_dir / "mirdb_mirtarget_standardized.tsv"
    targetscan_path.write_text(
        "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
        "ENSG1\t\t\thsa-miR-test\t0.9\n"
        "ENSG2\t\t\thsa-miR-test\t0.1\n"
        "ENSG3\t\t\thsa-miR-test\t0.2\n"
        "ENSG4\t\t\thsa-miR-test\t0.85\n"
        "ENSG5\t\t\thsa-miR-test\t0.15\n"
        "ENSG6\t\t\thsa-miR-test\t0.25\n",
        encoding="utf-8",
    )
    mirdb_path.write_text(
        "Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n"
        "ENSG1\t\t\thsa-miR-test\t0.8\n"
        "ENSG2\t\t\thsa-miR-test\t0.2\n"
        "ENSG3\t\t\thsa-miR-test\t0.3\n"
        "ENSG4\t\t\thsa-miR-test\t0.75\n"
        "ENSG5\t\t\thsa-miR-test\t0.25\n"
        "ENSG6\t\t\thsa-miR-test\t0.35\n",
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        [
            {
                "tool_id": "targetscan",
                "official_name": "TargetScan v8",
                "score_direction": "higher_is_stronger",
                "predictor_output_path": str(targetscan_path.resolve()),
            },
            {
                "tool_id": "mirdb_mirtarget",
                "official_name": "miRDB",
                "score_direction": "higher_is_stronger",
                "predictor_output_path": str(mirdb_path.resolve()),
            },
        ]
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
                "  id: [D001, D002]",
                "",
                "predictors:",
                "  tool_id: [targetscan, mirdb_mirtarget]",
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

    monkeypatch.setattr(benchmark, "sync_zenodo_experiments", lambda *args, **kwargs: [])

    benchmark.run_benchmark(config)

    summary_paths = list(out_root.glob("*/summary.json"))
    assert len(summary_paths) == 1
    out_dir = summary_paths[0].parent

    assert re.fullmatch(rf"{dt.date.today():%Y%m%d}_\d{{6}}", out_dir.name)
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "REPORT.pdf").is_file()
    assert (out_dir / "tables" / "per_experiment" / "coverage_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "positive_coverage_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "aps_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "pr_auc_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "per_experiment" / "auroc_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "combined" / "cross_dataset_predictor_summary.tsv").is_file()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["out_root"] == str(out_root)
    assert summary["out_dir"] == str(out_dir)
    assert summary["run_dir_name"] == out_dir.name
    assert summary["tags"] == ["demo", "end_to_end"]
    assert summary["metric_tables"]["auroc"].endswith("auroc_per_experiment.tsv")
    assert summary["readme"].endswith("README.md")
    assert summary["report_pdf"].endswith("REPORT.pdf")
    assert set(summary["dataset_ids"]) == {"D001", "D002"}
    assert summary["tool_ids"] == ["targetscan", "mirdb_mirtarget"]

    report_pdf = out_dir / "REPORT.pdf"
    run_media_boxes = _pdf_media_boxes(report_pdf)
    assert len(run_media_boxes) > 1
    assert len(set(run_media_boxes)) == 1
    run_width, run_height = run_media_boxes[0]
    assert run_width == pytest.approx(72.0 * benchmark.REPORT_PAGE_SIZE[0], abs=0.02)
    assert run_height == pytest.approx(72.0 * benchmark.REPORT_PAGE_SIZE[1], abs=0.02)

    dataset_pdf = (
        out_dir
        / "datasets"
        / "D001"
        / "reports"
        / "D001__targetscan_evaluation_report.pdf"
    )
    dataset_media_boxes = _pdf_media_boxes(dataset_pdf)
    assert len(dataset_media_boxes) >= 2
    assert len(set(dataset_media_boxes)) == 1

    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_coverage_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "metrics" / "cross_dataset_auroc_distribution.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_background_local_rank_distributions.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_background_local_rank_counts.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_background_global_rank_distributions.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_background_global_rank_counts.png"
    ).is_file()
    assert (
        out_dir / "plots" / "combined" / "ranks" / "positive_recovery_fraction_by_prediction_count.png"
    ).is_file()
    assert (
        out_dir
        / "plots"
        / "combined"
        / "combinations"
        / "predictor_combination_expanded_frontier.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "D001" / "plots" / "heatmaps" / "top_10pct_positive_genes.png"
    ).is_file()
    assert (
        out_dir / "datasets" / "D001" / "plots" / "comparisons" / "top_100_effect_cdfs.png"
    ).is_file()


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
        "ENSG1\tGENE1\t\thsa-miR-test\t0.9\n"
        "ENSG2\tGENE2\t\thsa-miR-test\t0.1\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tool_id": "tool_a",
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
                "  tool_id: [tool_a]",
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
            "ENSG1\t-2.0\t0.01\t0.001\n"
            "ENSG2\t0.2\t0.50\t0.500\n",
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
    assert joined["gene_id"].tolist() == ["ENSG1", "ENSG2"]
    assert joined["score_tool_a"].tolist() == [0.9, 0.1]


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
        "ENSG1\tGENE1\t\thsa-miR-test\t0.9\n"
        "ENSG2\tGENE2\t\thsa-miR-test\t0.1\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tool_id": "tool_a",
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
                "  tool_id: [tool_a]",
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
            "ENSG1\t-2.0\t0.01\t0.001\n"
            "ENSG2\t0.2\t0.50\t0.500\n",
            encoding="utf-8",
        )
        return [dest]

    captured = {}

    def fake_evaluate_joined_dataframe(joined, *args, **kwargs):
        joined["local_rank_tool_a"] = [0.8, 0.2]
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
    assert "local_rank_tool_a" in captured["joined_frames"][0].columns
