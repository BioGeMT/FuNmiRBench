"""Run the downstream benchmark from a single YAML config."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import yaml

from funmirbench import datasets
from funmirbench.cli.join_experiment_predictions import (
    build_combined_joined_dataset,
    load_predictions_registry,
)
from funmirbench.cli.plot_correlation import (
    evaluate_joined_dataset,
    write_metric_tables,
)
from funmirbench.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the downstream benchmark from canonical experiment and predictor artifacts."
    )
    parser.add_argument("--config", type=pathlib.Path, required=True)
    return parser.parse_args()


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_datasets(
    experiments_cfg: dict[str, Any],
    *,
    root: pathlib.Path,
    datasets_json: pathlib.Path,
) -> list[datasets.DatasetMeta]:
    dataset_ids = experiments_cfg.get("dataset_ids")
    filter_keys = ["miRNA", "cell_line", "perturbation", "tissue", "geo_accession"]
    has_filters = any(experiments_cfg.get(key) is not None for key in filter_keys)

    if dataset_ids is not None and has_filters:
        raise ValueError("Use either experiments.dataset_ids or experiment filters, not both.")

    if dataset_ids is not None:
        selected = [
            datasets.get_dataset(
                dataset_id,
                root=root,
                datasets_json=datasets_json,
            )
            for dataset_id in dataset_ids
        ]
        missing_ids = [
            dataset_id
            for dataset_id, meta in zip(dataset_ids, selected)
            if meta is None
        ]
        if missing_ids:
            raise ValueError(f"Unknown dataset ids: {missing_ids}")
        return [meta for meta in selected if meta is not None]

    selected = datasets.list_datasets(
        miRNA=experiments_cfg.get("miRNA"),
        cell_line=experiments_cfg.get("cell_line"),
        perturbation=experiments_cfg.get("perturbation"),
        tissue=experiments_cfg.get("tissue"),
        geo_accession=experiments_cfg.get("geo_accession"),
        root=root,
        datasets_json=datasets_json,
    )
    if not selected:
        raise ValueError("Experiment selection resolved to no datasets.")
    return selected


def run_benchmark(config_path: pathlib.Path) -> pathlib.Path:
    config_path = config_path.expanduser().resolve()
    config = _load_config(config_path)
    root = config_path.parent
    experiments_cfg = config["experiments"]
    predictors_cfg = config["predictors"]
    evaluation_cfg = config["evaluation"]
    datasets_json = root / "metadata" / "datasets.json"
    predictions_json = root / "metadata" / "predictions.json"
    out_dir = resolve_path(root, pathlib.Path(config.get("out_dir", "results/run_benchmark")))

    joined_dir = out_dir / "joined"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    tables_dir = out_dir / "tables"
    joined_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prediction_registry = load_predictions_registry(predictions_json)
    selected_datasets = _select_datasets(
        experiments_cfg,
        root=root,
        datasets_json=datasets_json,
    )
    tool_ids = predictors_cfg["tool_ids"]
    min_score = predictors_cfg.get("min_score")

    dataset_outputs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for meta in selected_datasets:
        joined, canonical_paths = build_combined_joined_dataset(
            meta,
            tool_ids=tool_ids,
            prediction_registry=prediction_registry,
            root=root,
            min_score=min_score,
        )
        joined_path = joined_dir / f"{meta.id}.tsv"
        joined.to_csv(joined_path, sep="\t", index=False)

        evaluation = evaluate_joined_dataset(
            joined_path,
            plots_dir=plots_dir,
            reports_dir=reports_dir,
            fdr_threshold=float(evaluation_cfg["fdr_threshold"]),
            abs_logfc_threshold=float(evaluation_cfg["abs_logfc_threshold"]),
            predictor_top_fraction=float(evaluation_cfg.get("predictor_top_fraction", 0.10)),
            dataset_id=meta.id,
            mirna=meta.miRNA,
            cell_line=meta.cell_line,
            perturbation=meta.perturbation,
            geo_accession=meta.geo_accession,
            de_table_path=str(meta.full_path),
            canonical_paths=canonical_paths,
        )
        metric_rows.extend(evaluation["metric_rows"])
        dataset_outputs.append(
            {
                "dataset_id": meta.id,
                "mirna": meta.miRNA,
                "cell_line": meta.cell_line,
                "perturbation": meta.perturbation,
                "geo_accession": meta.geo_accession,
                "de_table_path": str(meta.full_path),
                "joined_tsv": str(joined_path),
                "canonical_tsv_paths": canonical_paths,
                "plots": evaluation["plots"],
                "predictor_correlation_tsv": evaluation["predictor_correlation_tsv"],
            }
        )

    metric_tables = write_metric_tables(metric_rows, tables_dir)

    summary = {
        "root": str(root),
        "datasets_json": str(datasets_json),
        "predictions_json": str(predictions_json),
        "out_dir": str(out_dir),
        "dataset_ids": [meta.id for meta in selected_datasets],
        "experiment_selection": experiments_cfg,
        "tool_ids": tool_ids,
        "metric_tables": metric_tables,
        "datasets": dataset_outputs,
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        f"run_dir: {out_dir}",
        f"datasets_json: {datasets_json}",
        f"predictions_json: {predictions_json}",
        f"dataset_ids: {', '.join(meta.id for meta in selected_datasets)}",
        f"tool_ids: {', '.join(tool_ids)}",
        f"aps_table: {metric_tables['aps']}",
        f"spearman_table: {metric_tables['spearman']}",
        f"auroc_table: {metric_tables['auroc']}",
        "",
        "datasets:",
    ]
    for dataset_output in dataset_outputs:
        summary_lines.extend(
            [
                f"  - dataset_id: {dataset_output['dataset_id']}",
                f"    joined_tsv: {dataset_output['joined_tsv']}",
            ]
        )
        for tool_id in tool_ids:
            summary_lines.append(
                f"    report_{tool_id}: "
                f"{reports_dir / f'{dataset_output['dataset_id']}__{tool_id}_evaluation_report.txt'}"
            )
        if dataset_output["predictor_correlation_tsv"] is not None:
            summary_lines.append(
                f"    predictor_correlation_tsv: {dataset_output['predictor_correlation_tsv']}"
            )

    summary_txt = out_dir / "summary.txt"
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    args = parse_args()
    run_dir = run_benchmark(args.config)
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
