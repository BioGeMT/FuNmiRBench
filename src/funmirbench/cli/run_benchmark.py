"""Run the downstream FuNmiRBench benchmark from a single TOML config."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tomllib

from funmirbench import datasets
from funmirbench.cli import join_experiment_predictions, plot_correlation
from funmirbench.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the downstream benchmark from canonical experiment and predictor artifacts."
    )
    parser.add_argument("--config", type=pathlib.Path, required=True)
    return parser.parse_args()


def _run_cli(module, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv


def _load_config(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_predictions_registry(path: pathlib.Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["tool_id"]: entry for entry in entries}


def run_benchmark(config_path: pathlib.Path) -> pathlib.Path:
    config_path = config_path.expanduser().resolve()
    config = _load_config(config_path)
    config_dir = config_path.parent

    paths_cfg = config["paths"]
    experiments_cfg = config["experiments"]
    predictors_cfg = config["predictors"]
    evaluation_cfg = config["evaluation"]

    root = resolve_path(config_dir, pathlib.Path(paths_cfg["root"]))
    datasets_json = resolve_path(root, pathlib.Path(paths_cfg["datasets_json"]))
    predictions_json = resolve_path(root, pathlib.Path(paths_cfg["predictions_json"]))
    out_dir = resolve_path(root, pathlib.Path(paths_cfg["out_dir"]))

    joined_dir = out_dir / "joined"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    joined_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    prediction_registry = _load_predictions_registry(predictions_json)

    selected_datasets = [
        datasets.get_dataset(
            dataset_id,
            root=root,
            datasets_json=datasets_json,
        )
        for dataset_id in experiments_cfg["dataset_ids"]
    ]
    selected_tools = predictors_cfg["tool_ids"]

    min_score = predictors_cfg.get("min_score")
    score_col = evaluation_cfg.get("score_col")

    pairs = []

    for meta in selected_datasets:
        dataset_id = meta.id
        for tool_id in selected_tools:
            tool_meta = prediction_registry[tool_id]
            joined_out = joined_dir / f"{dataset_id}_{tool_id}.tsv"

            join_argv = [
                "join_experiment_predictions",
                "--dataset-id", dataset_id,
                "--tool", tool_id,
                "--predictions-json", str(predictions_json),
                "--root", str(root),
                "--out", str(joined_out),
            ]
            if min_score is not None:
                join_argv.extend(["--min-score", str(min_score)])
            _run_cli(join_experiment_predictions, join_argv)

            plot_argv = [
                "plot_correlation",
                "--joined-tsv", str(joined_out),
                "--out-dir", str(plots_dir),
                "--dataset-id", dataset_id,
                "--mirna", meta.miRNA,
                "--cell-line", meta.cell_line or "",
                "--perturbation", meta.perturbation or "",
                "--geo-accession", meta.geo_accession or "",
                "--top-n", str(evaluation_cfg["top_n"]),
                "--fdr-threshold", str(evaluation_cfg["fdr_threshold"]),
                "--abs-logfc-threshold", str(evaluation_cfg["abs_logfc_threshold"]),
            ]
            if score_col:
                plot_argv.extend(["--score-col", str(score_col)])
            _run_cli(plot_correlation, plot_argv)

            report_name = f"{joined_out.stem}_evaluation_report.txt"
            report_src = plots_dir / report_name
            report_dst = reports_dir / report_name
            shutil.move(str(report_src), str(report_dst))

            pairs.append(
                {
                    "dataset_id": dataset_id,
                    "tool_id": tool_id,
                    "mirna": meta.miRNA,
                    "cell_line": meta.cell_line,
                    "perturbation": meta.perturbation,
                    "geo_accession": meta.geo_accession,
                    "de_table_path": str(meta.full_path),
                    "canonical_tsv_path": tool_meta["canonical_tsv_path"],
                    "joined_tsv": str(joined_out),
                    "report_txt": str(report_dst),
                }
            )

    summary = {
        "root": str(root),
        "datasets_json": str(datasets_json),
        "predictions_json": str(predictions_json),
        "out_dir": str(out_dir),
        "dataset_ids": experiments_cfg["dataset_ids"],
        "tool_ids": selected_tools,
        "pairs": pairs,
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary_txt = out_dir / "summary.txt"
    lines = [
        f"run_dir: {out_dir}",
        f"datasets_json: {datasets_json}",
        f"predictions_json: {predictions_json}",
        f"dataset_ids: {', '.join(experiments_cfg['dataset_ids'])}",
        f"tool_ids: {', '.join(selected_tools)}",
        "",
        "pairs:",
    ]
    for pair in pairs:
        lines.extend(
            [
                f"  - dataset_id: {pair['dataset_id']}",
                f"    tool_id: {pair['tool_id']}",
                f"    joined_tsv: {pair['joined_tsv']}",
                f"    report_txt: {pair['report_txt']}",
            ]
        )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_dir


def main() -> None:
    args = parse_args()
    run_dir = run_benchmark(args.config)
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
