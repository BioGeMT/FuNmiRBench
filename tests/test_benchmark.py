"""End-to-end test via a small real-data benchmark config."""

import json
import pathlib
import subprocess
import sys

import pandas as pd

from funmirbench.build_cheating_predictions import build_cheating_scores
from funmirbench.build_predictions import build_mock_scores, write_tsv


def test_example_end_to_end(tmp_path):
    """Run a small two-predictor benchmark config and check outputs."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tmp_dir = tmp_path
    config = tmp_dir / "benchmark.yaml"
    out_dir = tmp_dir / "results"

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
    predictions = predictions[predictions["tool_id"].isin(["predictor_1", "predictor_2"])].copy()

    mock_path = tmp_dir / "mock_predictor_output.tsv"
    cheating_path = tmp_dir / "cheating_predictor_output.tsv"
    write_tsv(build_mock_scores(experiments_tsv, tmp_dir), mock_path)
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
            "predictor_1": str(mock_path.resolve()),
            "predictor_2": str(cheating_path.resolve()),
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
                "  tool_id: [predictor_1, predictor_2]",
                "",
                "evaluation:",
                "  fdr_threshold: 0.05",
                "  abs_logfc_threshold: 1.0",
                "  predictor_top_fraction: 0.10",
                "",
                f"out_dir: {out_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    stale_plot = out_dir / "plots" / "GSE109725_OE_miR_204_5p" / "mock_score_vs_logFC.png"
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_text("stale", encoding="utf-8")
    stale_report = out_dir / "reports" / "GSE109725_OE_miR_204_5p__mock_evaluation_report.txt"
    stale_report.parent.mkdir(parents=True, exist_ok=True)
    stale_report.write_text("stale", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "funmirbench.benchmark", "--config", str(config)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "tables" / "aps_per_experiment.tsv").is_file()
    assert (out_dir / "tables" / "pr_auc_per_experiment.tsv").is_file()

    joined_files = sorted((out_dir / "joined").glob("*.tsv"))
    assert [path.name for path in joined_files] == [
        "GSE109725_OE_miR_204_5p.tsv",
        "GSE118315_KO_miR_124_3p.tsv",
        "GSE210778_OE_miR_375_3p.tsv",
    ]

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["dataset_ids"]) == {
        "GSE109725_OE_miR_204_5p",
        "GSE210778_OE_miR_375_3p",
        "GSE118315_KO_miR_124_3p",
    }
    assert summary["tool_ids"] == ["predictor_1", "predictor_2"]

    plots = list((out_dir / "plots").rglob("*.png"))
    assert len(plots) == 18
    assert (
        out_dir / "plots" / "GSE109725_OE_miR_204_5p" / "predictor_1_score_vs_logFC.png"
    ).is_file()
    assert (
        out_dir / "plots" / "GSE109725_OE_miR_204_5p" / "predictor_pr_curves.png"
    ).is_file()
    assert (
        out_dir / "plots" / "GSE109725_OE_miR_204_5p" / "predictor_roc_curves.png"
    ).is_file()
    assert not (
        out_dir / "plots" / "GSE109725_OE_miR_204_5p" / "predictor_1_pr_curve.png"
    ).exists()
    assert not (
        out_dir / "plots" / "GSE109725_OE_miR_204_5p" / "predictor_2_roc_curve.png"
    ).exists()
    assert not stale_plot.exists()
    assert not stale_report.exists()

    aps_lines = (out_dir / "tables" / "aps_per_experiment.tsv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(aps_lines) == 4
    header = aps_lines[0].split("\t")
    assert "predictor_1" in header
    assert "predictor_2" in header
