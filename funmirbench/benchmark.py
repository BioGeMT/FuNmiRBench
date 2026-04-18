"""Run the FuNmiRBench benchmark from a single YAML config."""

import argparse
import hashlib
import json
import logging
import pathlib
import re
import shutil
import textwrap
import urllib.parse

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

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


def write_run_readme(
    out_dir,
    *,
    config_path,
    dataset_outputs,
    tool_ids,
    metric_tables,
    combined_outputs,
    fdr_threshold,
    abs_logfc_threshold,
    predictor_top_fraction,
):
    lines = [
        "# FuNmiRBench Run README",
        "",
        "## Summary",
        f"- Config: `{config_path}`",
        f"- Run directory: `{out_dir}`",
        f"- Datasets: `{len(dataset_outputs)}`",
        f"- Predictors: `{', '.join(tool_ids)}`",
        "",
        "## Evaluation Settings",
        (
            f"- GT positive threshold: `FDR < {fdr_threshold}` and perturbation-aware effect "
            f"`> {abs_logfc_threshold}` (`-logFC` for OE, `+logFC` for KO/KD)"
        ),
        f"- Predictor agreement top fraction: `{predictor_top_fraction:.0%}`",
        "- Score handling: predictors are first aligned so that higher always means stronger",
        "- Per-dataset heatmaps and agreement plots: dataset-local tie-aware dense ranking over scored rows",
        "- Cross-dataset rank-distribution plots: global tie-aware dense ranking over each predictor's full standardized file",
        "",
        "## Where To Start",
        "- Read `REPORT.pdf` for the main run-level summary with combined plots and explanatory notes",
        "- Open `tables/combined/cross_dataset_predictor_summary.tsv` for the compact numeric cross-dataset summary",
        "- Browse `plots/combined/` for cross-dataset comparison figures",
        "- Browse `datasets/<dataset_id>/` for dataset-specific tables, plots, and reports",
        "",
        "## Layout",
        "- `datasets/<dataset_id>/joined.tsv`: joined DE + predictor score table for one dataset",
        "- `datasets/<dataset_id>/plots/`: per-dataset visual outputs",
        "- `datasets/<dataset_id>/reports/`: per-dataset Markdown/PDF reports and correlation TSVs",
        "- `tables/per_experiment/`: metric tables across datasets, one row per experiment",
        "- `tables/combined/`: cross-dataset predictor summary table",
        "- `plots/combined/`: cross-dataset comparison plots",
        "- `summary.json`: machine-readable run summary",
        "",
        "## Datasets",
    ]
    for item in dataset_outputs:
        lines.extend(
            [
                (
                    f"- `{item['dataset_id']}`"
                    f" | miRNA `{item['mirna']}`"
                    f" | cell line `{item['cell_line']}`"
                    f" | joined `{item['joined_tsv']}`"
                )
            ]
        )
    lines.extend(
        [
            "",
            "## Key Outputs",
            f"- per-experiment tables: `{metric_tables}`",
            f"- combined outputs: `{combined_outputs}`",
            "",
        ]
    )
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def write_run_pdf_report(
    out_dir,
    *,
    config_path,
    dataset_outputs,
    tool_ids,
    metric_tables,
    combined_outputs,
    fdr_threshold,
    abs_logfc_threshold,
    predictor_top_fraction,
):
    report_path = out_dir / "REPORT.pdf"
    text_lines = [
        "# FuNmiRBench Run Report",
        "",
        "## Run Summary",
        f"- Config: {config_path}",
        f"- Run directory: {out_dir}",
        f"- Datasets: {len(dataset_outputs)}",
        f"- Predictors: {', '.join(tool_ids)}",
        "",
        "## Evaluation Settings",
        (
            f"- GT positives were defined as FDR < {fdr_threshold} and perturbation-aware effect "
            f"> {abs_logfc_threshold} (-logFC for OE, +logFC for KO/KD)"
        ),
        f"- Predictor-correlation top fraction: {predictor_top_fraction:.0%}",
        "- Predictor scores were aligned to a common higher-is-stronger direction before evaluation",
        "- Per-dataset heatmaps and agreement plots use a dataset-local tie-aware dense ranking over scored rows",
        "- Cross-dataset rank-distribution plots use a global tie-aware dense ranking computed over each predictor's full standardized file",
        "",
        "## Output Guide",
        f"- Per-experiment metric tables: {metric_tables}",
        f"- Cross-dataset outputs: {combined_outputs}",
        "- Dataset-specific outputs live under datasets/<dataset_id>/",
        "",
        "## Datasets",
    ]
    for item in dataset_outputs:
        text_lines.append(
            f"- {item['dataset_id']} | miRNA {item['mirna']} | cell line {item['cell_line']} | perturbation {item['perturbation']}"
        )

    style_map = {
        "h1": {"fontsize": 17, "weight": "bold", "color": "#17324D", "gap": 0.060},
        "h2": {"fontsize": 12, "weight": "bold", "color": "#2F5D8C", "gap": 0.042},
        "body": {"fontsize": 9.5, "weight": "normal", "color": "#22303C", "gap": 0.026},
        "blank": {"fontsize": 9.5, "weight": "normal", "color": "#22303C", "gap": 0.018},
    }

    def iter_lines():
        for raw_line in text_lines:
            if raw_line.startswith("# "):
                yield {"text": raw_line[2:], "kind": "h1"}
                continue
            if raw_line.startswith("## "):
                yield {"text": raw_line[3:], "kind": "h2"}
                continue
            if raw_line.startswith("- "):
                wrapped = textwrap.wrap(raw_line[2:], width=92) or [""]
                for index, chunk in enumerate(wrapped):
                    prefix = "- " if index == 0 else "  "
                    yield {"text": prefix + chunk, "kind": "body"}
                continue
            if not raw_line.strip():
                yield {"text": "", "kind": "blank"}
                continue
            for chunk in textwrap.wrap(raw_line, width=94) or [""]:
                yield {"text": chunk, "kind": "body"}

    with PdfPages(report_path) as pdf:
        fig = None
        ax = None
        y = 0.0

        def new_text_page():
            page_fig, page_ax = plt.subplots(figsize=(8.27, 11.69))
            page_ax.axis("off")
            page_fig.patch.set_facecolor("white")
            return page_fig, page_ax, 0.95

        for item in iter_lines():
            style = style_map[item["kind"]]
            if fig is None or y - style["gap"] < 0.06:
                if fig is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                fig, ax, y = new_text_page()
            if item["text"]:
                ax.text(
                    0.06,
                    y,
                    item["text"],
                    fontsize=style["fontsize"],
                    fontweight=style["weight"],
                    color=style["color"],
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                )
            y -= style["gap"]

        if fig is None:
            fig, ax, y = new_text_page()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        plot_items = []
        for key in [
            "cross_dataset_metric_heatmap",
            "cross_dataset_metric_distributions",
            "coverage_vs_performance",
            "positive_background_rank_distributions",
        ]:
            path = combined_outputs.get("plots", {}).get(key)
            if path:
                plot_items.append((key, pathlib.Path(path)))

        for key, path in plot_items:
            if not path.is_file():
                continue
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            fig.patch.set_facecolor("white")
            ax.text(
                0.06,
                0.975,
                key.replace("_", " ").title(),
                fontsize=12,
                fontweight="bold",
                color="#17324D",
                va="top",
                ha="left",
            )
            ax.imshow(image)
            ax.set_position([0.08, 0.08, 0.84, 0.82])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return report_path


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

    datasets_dir = out_dir / "datasets"
    plots_dir = out_dir / "plots"
    combined_plots_dir = plots_dir / "combined"
    tables_dir = out_dir / "tables"
    per_experiment_tables_dir = tables_dir / "per_experiment"
    combined_tables_dir = tables_dir / "combined"
    for path in (
        datasets_dir,
        plots_dir,
        combined_plots_dir,
        tables_dir,
        per_experiment_tables_dir,
        combined_tables_dir,
    ):
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
        dataset_dir = datasets_dir / meta.id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_plots_dir = dataset_dir / "plots"
        dataset_reports_dir = dataset_dir / "reports"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_plots_dir.mkdir(parents=True, exist_ok=True)
        dataset_reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Joining predictions for {meta.id}...")
        joined, predictor_output_paths = build_joined(meta, tool_ids, predictions, root)
        joined_path = dataset_dir / "joined.tsv"
        joined_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(joined_path, sep="\t", index=False)
        logger.info(f"  Wrote joined table: {joined_path}")
        joined_frames.append(joined.copy())

        logger.info(f"  Evaluating metrics and plots for {meta.id}...")
        evaluation = evaluate_joined_dataframe(
            joined,
            plots_dir=dataset_plots_dir,
            reports_dir=dataset_reports_dir,
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
                "dataset_dir": str(dataset_dir),
                "predictor_output_paths": predictor_output_paths,
                "plots": evaluation["plots"],
                "predictor_correlation_tsv": evaluation["predictor_correlation_tsv"],
            }
        )
        logger.info(f"  Finished {meta.id}")

    logger.info("Writing metric tables...")
    metric_tables = write_metric_tables(metric_rows, per_experiment_tables_dir, logger=logger.info)
    logger.info("Writing cross-dataset summaries...")
    combined_outputs = write_cross_dataset_summaries(
        metric_rows,
        combined_tables_dir,
        combined_plots_dir,
        joined_frames=joined_frames,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        logger=logger.info,
    )
    readme_path = write_run_readme(
        out_dir,
        config_path=config_path,
        dataset_outputs=dataset_outputs,
        tool_ids=tool_ids,
        metric_tables=metric_tables,
        combined_outputs=combined_outputs,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=float(eval_cfg.get("predictor_top_fraction", 0.10)),
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
        predictor_top_fraction=float(eval_cfg.get("predictor_top_fraction", 0.10)),
    )

    summary = {
        "config": str(config_path),
        "out_root": str(out_root),
        "out_dir": str(out_dir),
        "run_dir_name": out_dir.name,
        "tags": config.get("tags", []),
        "dataset_ids": [meta.id for meta in experiments],
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
    return out_dir


def main():
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))
    out_dir = run_benchmark(args.config)
    logger.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
