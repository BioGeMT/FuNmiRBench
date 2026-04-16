"""Run the FuNmiRBench benchmark from a single YAML config."""

import argparse
import hashlib
import json
import logging
import pathlib
import re
import shutil
import urllib.parse

import pandas as pd
import yaml

from funmirbench import DatasetMeta
from funmirbench.evaluate import (
    evaluate_joined_dataframe,
    write_cross_dataset_summaries,
    write_metric_tables,
)
from funmirbench.experiment_store import sync_zenodo_experiments
from funmirbench.join import build_joined
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)


def _slugify(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "na"


def _summarize_values(prefix, values, *, max_items=2):
    items = [_slugify(value) for value in values if str(value).strip()]
    if not items:
        return f"{prefix}-none"

    digest = hashlib.sha1("|".join(items).encode("utf-8")).hexdigest()[:8]
    if len(items) <= max_items:
        body = "-".join(items)
    else:
        body = "-".join(items[:max_items]) + f"-plus{len(items) - max_items}"

    part = f"{prefix}-{body}"
    if len(part) > 80:
        return f"{prefix}-{len(items)}items-{digest}"
    return part


def build_run_dir_name(*, experiments, tool_ids, eval_cfg, tags=None):
    del eval_cfg
    parts = []

    if tags:
        if isinstance(tags, str):
            tags = [tags]
        parts.append(_summarize_values("tag", tags, max_items=3))

    perturbations = {
        str(meta.perturbation).strip().upper()
        for meta in experiments
        if str(meta.perturbation).strip()
    }
    cell_lines = {
        str(meta.cell_line).strip()
        for meta in experiments
        if str(meta.cell_line).strip() and str(meta.cell_line).strip().upper() != "NA"
    }

    parts.extend(
        [
            f"exp{len(experiments)}",
            f"pred{len(tool_ids)}",
            f"oe{int('OE' in perturbations)}",
            f"ko{int('KO' in perturbations)}",
            f"kd{int('KD' in perturbations)}",
            f"cell{len(cell_lines)}",
        ]
    )
    return "__".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FuNmiRBench benchmark.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def filter_df(df, filters):
    """AND across columns, OR within each column's value list."""
    for col, values in filters.items():
        if not isinstance(values, list):
            values = [values]
        df = df[df[col].isin(values)]
    return df


def load_experiments(tsv_path, root, filters):
    df = pd.read_csv(tsv_path, sep="\t")
    if filters:
        df = filter_df(df, filters)

    metas = []
    for _, row in df.iterrows():
        parsed = urllib.parse.urlparse(str(row.get("gse_url", "") or ""))
        geo = urllib.parse.parse_qs(parsed.query).get("acc", [None])[0]
        metas.append(
            DatasetMeta(
                id=str(row["id"]),
                miRNA=str(row["mirna_name"]),
                cell_line=str(row.get("tested_cell_line", "") or ""),
                tissue=str(row.get("tissue", "") or ""),
                perturbation=str(row.get("experiment_type", "") or ""),
                organism=str(row.get("organism", "") or ""),
                geo_accession=geo,
                data_path=str(row["de_table_path"]),
                root=root,
            )
        )
    return metas


def selected_experiment_paths(tsv_path, filters) -> list[str]:
    df = pd.read_csv(tsv_path, sep="\t")
    if filters:
        df = filter_df(df, filters)
    return [str(value) for value in df["de_table_path"].tolist()]


def load_predictions(tsv_path, filters):
    df = pd.read_csv(tsv_path, sep="\t")
    if filters:
        df = filter_df(df, filters)
    if df["tool_id"].duplicated().any():
        raise ValueError("Duplicate tool_id values found after predictor filtering.")
    return {row["tool_id"]: row.to_dict() for _, row in df.iterrows()}


def clear_dataset_outputs(dataset_id, plots_dir, reports_dir):
    dataset_plots_dir = plots_dir / dataset_id
    if dataset_plots_dir.exists():
        shutil.rmtree(dataset_plots_dir)

    for stale_report in reports_dir.glob(f"{dataset_id}__*"):
        if stale_report.is_file():
            stale_report.unlink()


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

    joined_dir = out_dir / "joined"
    plots_dir = out_dir / "plots"
    combined_plots_dir = plots_dir / "combined"
    reports_dir = out_dir / "reports"
    tables_dir = out_dir / "tables"
    for path in (joined_dir, plots_dir, combined_plots_dir, reports_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)

    tool_ids = list(predictions)
    metric_rows = []
    dataset_outputs = []
    joined_frames = []
    fdr_threshold = float(eval_cfg.get("fdr_threshold", 0.05))
    abs_logfc_threshold = float(eval_cfg.get("abs_logfc_threshold", 1.0))

    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Predictors:  {tool_ids}")

    for meta in experiments:
        logger.info(f"Dataset: {meta.id} | {meta.miRNA} | {meta.cell_line}")
        clear_dataset_outputs(meta.id, plots_dir, reports_dir)
        logger.info(f"  Joining predictions for {meta.id}...")
        joined, predictor_output_paths = build_joined(meta, tool_ids, predictions, root)
        joined_path = joined_dir / f"{meta.id}.tsv"
        joined_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(joined_path, sep="\t", index=False)
        logger.info(f"  Wrote joined table: {joined_path}")
        joined_frames.append(joined.copy())

        logger.info(f"  Evaluating metrics and plots for {meta.id}...")
        evaluation = evaluate_joined_dataframe(
            joined,
            plots_dir=plots_dir,
            reports_dir=reports_dir,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            predictor_top_fraction=float(eval_cfg.get("predictor_top_fraction", 0.10)),
            dataset_id=meta.id,
            mirna=meta.miRNA,
            cell_line=meta.cell_line,
            perturbation=meta.perturbation,
            geo_accession=meta.geo_accession,
            de_table_path=str(meta.full_path),
            joined_tsv=joined_path,
            predictor_output_paths=predictor_output_paths,
            logger=logger.info,
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
                "predictor_output_paths": predictor_output_paths,
                "plots": evaluation["plots"],
                "predictor_correlation_tsv": evaluation["predictor_correlation_tsv"],
            }
        )
        logger.info(f"  Finished {meta.id}")

    logger.info("Writing metric tables...")
    metric_tables = write_metric_tables(metric_rows, tables_dir, logger=logger.info)
    logger.info("Writing cross-dataset summaries...")
    combined_outputs = write_cross_dataset_summaries(
        metric_rows,
        tables_dir,
        combined_plots_dir,
        joined_frames=joined_frames,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        logger=logger.info,
    )

    summary = {
        "config": str(config_path),
        "out_root": str(out_root),
        "out_dir": str(out_dir),
        "run_dir_name": out_dir.name,
        "tags": config.get("tags", []),
        "dataset_ids": [meta.id for meta in experiments],
        "tool_ids": tool_ids,
        "metric_tables": metric_tables,
        "cross_dataset_outputs": combined_outputs,
        "datasets": dataset_outputs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote summary: {summary_path}")
    return out_dir


def main():
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))
    out_dir = run_benchmark(args.config)
    logger.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
