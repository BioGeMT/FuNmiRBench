"""Run the FuNmiRBench benchmark from a single YAML config."""

import argparse
import json
import pathlib
import shutil
import urllib.parse

import pandas as pd
import yaml

from funmirbench import DatasetMeta
from funmirbench.evaluate import evaluate_joined_dataframe, write_metric_tables
from funmirbench.join import build_joined


def log(message):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FuNmiRBench benchmark.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
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
    log(f"Config: {config_path}")
    log("Loading benchmark config...")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parent

    log("Loading experiments...")
    experiments = load_experiments(
        root / config["experiments_tsv"],
        root,
        config.get("experiments"),
    )
    if not experiments:
        raise ValueError("Experiment selection resolved to no datasets.")

    log("Loading predictors...")
    predictions = load_predictions(
        root / config["predictions_tsv"],
        config.get("predictors"),
    )
    if not predictions:
        raise ValueError("Predictor selection resolved to no predictors.")

    eval_cfg = config.get("evaluation", {})
    out_dir = (root / config.get("out_dir", "results")).resolve()

    joined_dir = out_dir / "joined"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    tables_dir = out_dir / "tables"
    for path in (joined_dir, plots_dir, reports_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)

    tool_ids = list(predictions)
    metric_rows = []
    dataset_outputs = []

    log(f"Experiments: {len(experiments)}")
    log(f"Predictors:  {tool_ids}")

    for meta in experiments:
        log(f"Dataset: {meta.id} | {meta.miRNA} | {meta.cell_line}")
        clear_dataset_outputs(meta.id, plots_dir, reports_dir)
        log(f"  Joining predictions for {meta.id}...")
        joined, canonical_paths = build_joined(meta, tool_ids, predictions, root)
        joined_path = joined_dir / f"{meta.id}.tsv"
        joined.to_csv(joined_path, sep="\t", index=False)
        log(f"  Wrote joined table: {joined_path}")

        log(f"  Evaluating metrics and plots for {meta.id}...")
        evaluation = evaluate_joined_dataframe(
            joined,
            plots_dir=plots_dir,
            reports_dir=reports_dir,
            fdr_threshold=float(eval_cfg.get("fdr_threshold", 0.05)),
            abs_logfc_threshold=float(eval_cfg.get("abs_logfc_threshold", 1.0)),
            predictor_top_fraction=float(eval_cfg.get("predictor_top_fraction", 0.10)),
            dataset_id=meta.id,
            mirna=meta.miRNA,
            cell_line=meta.cell_line,
            perturbation=meta.perturbation,
            geo_accession=meta.geo_accession,
            de_table_path=str(meta.full_path),
            joined_tsv=joined_path,
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
        log(f"  Finished {meta.id}")

    log("Writing metric tables...")
    metric_tables = write_metric_tables(metric_rows, tables_dir)

    summary = {
        "config": str(config_path),
        "out_dir": str(out_dir),
        "dataset_ids": [meta.id for meta in experiments],
        "tool_ids": tool_ids,
        "metric_tables": metric_tables,
        "datasets": dataset_outputs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote summary: {summary_path}")
    return out_dir


def main():
    args = parse_args()
    out_dir = run_benchmark(args.config)
    log(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
