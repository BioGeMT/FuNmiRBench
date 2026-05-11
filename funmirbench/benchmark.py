"""Run the FuNmiRBench benchmark from a single YAML config."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil

import yaml

import funmirbench.evaluate as evaluate_module
from funmirbench.benchmark_config import (
    DEFAULT_DEMO_ABS_LOGFC_THRESHOLD,
    DEFAULT_DEMO_FDR_THRESHOLD,
    THRESHOLD_SENSITIVE_DEMO_TOOLS,
    build_run_dir_name,
    filter_df,
    load_experiments,
    load_predictions,
    selected_experiment_paths,
    validate_threshold_sensitive_predictors,
)
from funmirbench.benchmark_reports import (
    _init_run_layout,
    write_run_readme,
)
from funmirbench.common_predictions import (
    write_combined_common_prediction_summary,
    write_common_prediction_summary,
)
from funmirbench.comparison_plots import write_common_comparison_plots
from funmirbench.cross_dataset import write_cross_dataset_summaries, write_metric_tables
from funmirbench.dataset_reports import write_predictor_reports
from funmirbench.evaluate import (
    REPORT_PAGE_SIZE,
    evaluate_joined_dataframe,
)
from funmirbench.experiment_store import sync_zenodo_experiments
from funmirbench.join import build_joined
from funmirbench.logger import parse_log_level, setup_logging
from funmirbench.predictor_combinations import write_predictor_combination_outputs
from funmirbench.run_report import write_run_pdf_report


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FuNmiRBench benchmark.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def clear_dataset_outputs(dataset_id, plots_dir, reports_dir):
    dataset_plots_dir = plots_dir / dataset_id
    if dataset_plots_dir.exists():
        shutil.rmtree(dataset_plots_dir)

    for stale_report in reports_dir.glob(f"{dataset_id}__*"):
        if stale_report.is_file():
            stale_report.unlink()


def _finalize_run_bundle(
    out_dir,
    *,
    out_root,
    config_path,
    tags,
    dataset_outputs,
    tool_ids,
    metric_rows,
    joined_frames,
    common_prediction_summaries,
    tool_labels,
    fdr_threshold,
    abs_logfc_threshold,
    predictor_top_fraction,
):
    layout = _init_run_layout(out_dir)
    metric_tables = write_metric_tables(
        metric_rows,
        layout["per_experiment_tables_dir"],
        logger=logger.info,
    )
    combined_outputs = write_cross_dataset_summaries(
        metric_rows,
        layout["combined_tables_dir"],
        layout["combined_plots_dir"],
        joined_frames=joined_frames,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        tool_labels=tool_labels,
        logger=logger.info,
    )
    common_summary_path = write_combined_common_prediction_summary(
        common_prediction_summaries,
        layout["combined_tables_dir"],
    )
    combined_outputs.setdefault("tables", {})["common_prediction_summary"] = str(common_summary_path)
    combination_outputs = write_predictor_combination_outputs(
        joined_frames,
        layout["combined_tables_dir"],
        layout["combined_plots_dir"],
        tool_ids=tool_ids,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        logger=logger.info,
    )
    combined_outputs.setdefault("tables", {}).update(combination_outputs.get("tables", {}))
    combined_outputs.setdefault("plots", {}).update(combination_outputs.get("plots", {}))
    readme_path = write_run_readme(
        out_dir,
        config_path=config_path,
        dataset_outputs=dataset_outputs,
        tool_ids=tool_ids,
        metric_tables=metric_tables,
        combined_outputs=combined_outputs,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
    )
    report_path = write_run_pdf_report(
        out_dir,
        config_path=config_path,
        dataset_outputs=dataset_outputs,
        tool_ids=tool_ids,
        metric_tables=metric_tables,
        combined_outputs=combined_outputs,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
    )
    summary = {
        "config": str(config_path),
        "out_root": str(out_root),
        "out_dir": str(out_dir),
        "run_dir_name": out_dir.name,
        "tags": tags or [],
        "dataset_ids": [item["dataset_id"] for item in dataset_outputs],
        "tool_ids": tool_ids,
        "readme": str(readme_path),
        "report_pdf": str(report_path),
        "metric_tables": metric_tables,
        "cross_dataset_outputs": combined_outputs,
        "datasets": dataset_outputs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote summary: {summary_path}")
    return {
        "metric_tables": metric_tables,
        "combined_outputs": combined_outputs,
        "readme_path": readme_path,
        "report_path": report_path,
        "summary": summary,
        "summary_path": summary_path,
    }


def run_benchmark(config_path):
    config_path = config_path.expanduser().resolve()
    logger.info(f"Config: {config_path}")
    logger.info("Loading benchmark config...")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parent

    logger.info("Syncing selected experiment DE tables from Zenodo...")
    synced = sync_zenodo_experiments(
        selected_experiment_paths(
            root / config["experiments_tsv"],
            config.get("experiments"),
        ),
        repo=root,
    )
    logger.info(f"Synced {len(synced)} experiment DE tables.")

    logger.info("Loading experiments...")
    experiments = load_experiments(
        root / config["experiments_tsv"],
        root,
        config.get("experiments"),
    )
    if not experiments:
        raise ValueError("Experiment selection resolved to no datasets.")

    logger.info("Loading predictors...")
    predictions = load_predictions(
        root / config["predictions_tsv"],
        config.get("predictors"),
    )
    if not predictions:
        raise ValueError("Predictor selection resolved to no predictors.")

    eval_cfg = config.get("evaluation", {})
    evaluate_module.FIGURE_DPI = int(eval_cfg.get("figure_dpi", eval_cfg.get("publication_figure_dpi", 450)))
    out_root = (root / config.get("out_dir", "results")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir_name = build_run_dir_name(
        experiments=experiments,
        tool_ids=list(predictions),
        eval_cfg=eval_cfg,
        tags=config.get("tags"),
    )
    out_dir = out_root / run_dir_name
    suffix = 2
    while out_dir.exists():
        out_dir = out_root / f"{run_dir_name}__r{suffix}"
        suffix += 1
    logger.info(f"Results root: {out_root}")
    logger.info(f"Run output dir: {out_dir}")
    main_layout = _init_run_layout(out_dir)

    tool_ids = list(predictions)
    tool_labels = {
        str(tool_id): str(meta.get("official_name") or tool_id)
        for tool_id, meta in predictions.items()
    }
    metric_rows = []
    dataset_outputs = []
    joined_frames = []
    common_prediction_summaries = []
    fdr_threshold = float(eval_cfg.get("fdr_threshold", 0.05))
    abs_logfc_threshold = float(eval_cfg.get("abs_logfc_threshold", 1.0))
    predictor_top_fraction = float(eval_cfg.get("predictor_top_fraction", 0.10))
    write_top_prediction_cdfs = bool(eval_cfg.get("write_top_prediction_cdfs", True))
    report_min_common_coverage = float(eval_cfg.get("report_min_common_coverage", eval_cfg.get("publication_min_common_coverage", 0.10)))
    validate_threshold_sensitive_predictors(
        predictions,
        root=root,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    )

    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Predictors:  {tool_ids}")

    for meta in experiments:
        logger.info(f"Dataset: {meta.id} | {meta.miRNA} | {meta.cell_line}")
        dataset_dir = main_layout["datasets_dir"] / meta.id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        (dataset_dir / "plots").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "reports").mkdir(parents=True, exist_ok=True)
        logger.info(f"  Joining predictions for {meta.id}...")
        joined, predictor_output_paths = build_joined(meta, tool_ids, predictions, root)
        joined_path = dataset_dir / "joined.tsv"
        joined.to_csv(joined_path, sep="\t", index=False)
        logger.info(f"  Wrote joined table: {joined_path}")

        logger.info(f"  Evaluating metrics and plots for {meta.id}...")
        evaluation = evaluate_joined_dataframe(
            joined,
            plots_dir=dataset_dir / "plots",
            reports_dir=dataset_dir / "reports",
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            predictor_top_fraction=predictor_top_fraction,
            dataset_id=meta.id,
            mirna=meta.miRNA,
            cell_line=meta.cell_line,
            perturbation=meta.perturbation,
            geo_accession=meta.geo_accession,
            de_table_path=str(meta.full_path),
            joined_tsv=joined_path,
            predictor_output_paths=predictor_output_paths,
            tool_labels=tool_labels,
            write_top_prediction_cdfs=write_top_prediction_cdfs,
            logger=logger.info,
        )
        write_common_comparison_plots(
            joined,
            evaluation=evaluation,
            dataset_metric_rows=evaluation["metric_rows"],
            plots_dir=dataset_dir / "plots",
            dataset_id=meta.id,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            perturbation=meta.perturbation,
            min_common_coverage=report_min_common_coverage,
            logger=logger.info,
        )
        common_summary_path, common_prediction_summary = write_common_prediction_summary(
            joined,
            dataset_dir / "reports",
            dataset_id=meta.id,
            tool_ids=tool_ids,
            report_min_common_coverage=report_min_common_coverage,
        )
        common_prediction_summaries.append(common_prediction_summary)
        write_predictor_reports(
            reports_dir=dataset_dir / "reports",
            plots_dir=dataset_dir / "plots",
            dataset_id=meta.id,
            mirna=meta.miRNA,
            cell_line=meta.cell_line,
            perturbation=meta.perturbation,
            geo_accession=meta.geo_accession,
            de_table_path=str(meta.full_path),
            predictor_output_paths=predictor_output_paths,
            metric_rows=evaluation["metric_rows"],
            tool_labels=tool_labels,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            common_prediction_summary=common_prediction_summary,
        )
        joined_frames.append(joined.copy())
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
                "dataset_dir": str(dataset_dir),
                "predictor_output_paths": predictor_output_paths,
                "plots": evaluation["plots"],
                "common_prediction_summary_tsv": str(common_summary_path),
            }
        )
        logger.info(f"  Finished {meta.id}")

    logger.info("Writing metric tables...")
    logger.info("Writing cross-dataset summaries...")
    _finalize_run_bundle(
        out_dir,
        out_root=out_root,
        config_path=config_path,
        tags=config.get("tags"),
        dataset_outputs=dataset_outputs,
        tool_ids=tool_ids,
        metric_rows=metric_rows,
        joined_frames=joined_frames,
        common_prediction_summaries=common_prediction_summaries,
        tool_labels=tool_labels,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
    )
    return out_dir


def main():
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))
    out_dir = run_benchmark(args.config)
    logger.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
