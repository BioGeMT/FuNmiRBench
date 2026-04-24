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
    describe_gt_rule,
    evaluate_joined_dataframe,
    write_cross_dataset_summaries,
    write_metric_tables,
)
from funmirbench.experiment_store import sync_zenodo_experiments
from funmirbench.join import build_joined
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)
DEFAULT_DEMO_FDR_THRESHOLD = 0.05
DEFAULT_DEMO_ABS_LOGFC_THRESHOLD = 1.0
THRESHOLD_SENSITIVE_DEMO_TOOLS = {"cheating", "perfect"}
NO_FDR_COMPARISON_DIRNAME = "no_fdr_threshold"


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


def _resolve_predictor_output_path(root, predictor_output_path):
    path = pathlib.Path(predictor_output_path)
    if not path.is_absolute():
        path = root / path
    return path


def _predictor_metadata_sidecar_path(predictor_output_path):
    return predictor_output_path.with_suffix(predictor_output_path.suffix + ".meta.json")


def _thresholds_match(left, right, *, atol=1e-12):
    return abs(float(left) - float(right)) <= atol


def validate_threshold_sensitive_predictors(predictions, *, root, fdr_threshold, abs_logfc_threshold):
    for tool_id, tool_meta in predictions.items():
        if tool_id not in THRESHOLD_SENSITIVE_DEMO_TOOLS:
            continue

        output_path = _resolve_predictor_output_path(root, tool_meta["predictor_output_path"])
        metadata_path = _predictor_metadata_sidecar_path(output_path)
        thresholds_are_default = (
            _thresholds_match(fdr_threshold, DEFAULT_DEMO_FDR_THRESHOLD)
            and _thresholds_match(abs_logfc_threshold, DEFAULT_DEMO_ABS_LOGFC_THRESHOLD)
        )

        if not metadata_path.is_file():
            if thresholds_are_default:
                continue
            raise ValueError(
                "Selected threshold-sensitive demo predictor "
                f"{tool_id!r} at {output_path} has no sidecar metadata file "
                f"({metadata_path}). Regenerate it with matching thresholds before benchmarking."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        built_fdr_threshold = metadata.get("fdr_threshold")
        built_abs_logfc_threshold = metadata.get("abs_logfc_threshold")
        if built_fdr_threshold is None or built_abs_logfc_threshold is None:
            raise ValueError(
                "Threshold-sensitive demo predictor "
                f"{tool_id!r} metadata file {metadata_path} is missing build threshold fields."
            )
        if not (
            _thresholds_match(fdr_threshold, built_fdr_threshold)
            and _thresholds_match(abs_logfc_threshold, built_abs_logfc_threshold)
        ):
            raise ValueError(
                "Selected threshold-sensitive demo predictor "
                f"{tool_id!r} was built with thresholds "
                f"FDR<{built_fdr_threshold} and effect>{built_abs_logfc_threshold}, "
                f"but the benchmark is configured for FDR<{fdr_threshold} and effect>{abs_logfc_threshold}. "
                f"Regenerate {output_path.name} with matching thresholds."
            )


def clear_dataset_outputs(dataset_id, plots_dir, reports_dir):
    dataset_plots_dir = plots_dir / dataset_id
    if dataset_plots_dir.exists():
        shutil.rmtree(dataset_plots_dir)

    for stale_report in reports_dir.glob(f"{dataset_id}__*"):
        if stale_report.is_file():
            stale_report.unlink()


def _relative_display_path(path, *, base_dir):
    resolved = pathlib.Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(base_dir.resolve()))
    except ValueError:
        return str(resolved)


def _format_summary_value(value, *, percent=False):
    if pd.isna(value):
        return "NA"
    number = float(value)
    if percent:
        return f"{number:.1%}"
    return f"{number:.3f}"


def _gt_threshold_box_text(fdr_threshold, abs_logfc_threshold):
    if fdr_threshold is None:
        return f"no FDR\neffect > {float(abs_logfc_threshold)}"
    return f"FDR < {float(fdr_threshold)}\neffect > {float(abs_logfc_threshold)}"


def _init_run_layout(base_dir):
    datasets_dir = base_dir / "datasets"
    plots_dir = base_dir / "plots"
    combined_plots_dir = plots_dir / "combined"
    tables_dir = base_dir / "tables"
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
    return {
        "datasets_dir": datasets_dir,
        "plots_dir": plots_dir,
        "combined_plots_dir": combined_plots_dir,
        "tables_dir": tables_dir,
        "per_experiment_tables_dir": per_experiment_tables_dir,
        "combined_tables_dir": combined_tables_dir,
    }


def _load_cross_dataset_summary(combined_outputs):
    summary_path = combined_outputs.get("tables", {}).get("cross_dataset_predictor_summary")
    if not summary_path:
        return None
    summary_file = pathlib.Path(summary_path)
    if not summary_file.is_file():
        return None
    summary_df = pd.read_csv(summary_file, sep="\t")
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        ["aps_mean", "auroc_mean", "positive_coverage_mean", "coverage_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _cross_dataset_markdown_table(summary_df):
    lines = [
        "| Predictor | Mean coverage | Mean positive coverage | Mean APS | Mean PR-AUC | Mean Spearman | Mean AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.tool_id),
                    _format_summary_value(row.coverage_mean, percent=True),
                    _format_summary_value(row.positive_coverage_mean, percent=True),
                    _format_summary_value(row.aps_mean),
                    _format_summary_value(row.pr_auc_mean),
                    _format_summary_value(row.spearman_mean),
                    _format_summary_value(row.auroc_mean),
                ]
            )
            + " |"
        )
    return lines


def _report_takeaways(summary_df):
    if summary_df is None or summary_df.empty:
        return []
    takeaways = []
    specs = [
        ("aps_mean", "Highest mean APS"),
        ("auroc_mean", "Highest mean AUROC"),
        ("positive_coverage_mean", "Highest mean positive coverage"),
        ("spearman_mean", "Highest mean Spearman"),
    ]
    for metric_col, label in specs:
        row = summary_df.sort_values(metric_col, ascending=False).iloc[0]
        formatted = _format_summary_value(
            row[metric_col],
            percent=metric_col.endswith("coverage_mean"),
        )
        takeaways.append(f"{label}: {row['tool_id']} ({formatted})")
    sparse_row = summary_df.sort_values("coverage_mean", ascending=True).iloc[0]
    takeaways.append(
        "Lowest mean overall coverage: "
        f"{sparse_row['tool_id']} ({_format_summary_value(sparse_row['coverage_mean'], percent=True)})"
    )
    return takeaways


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
    comparison_runs=None,
):
    summary_df = _load_cross_dataset_summary(combined_outputs)
    relative_metric_tables = {
        key: _relative_display_path(path, base_dir=out_dir)
        for key, path in metric_tables.items()
    }
    relative_combined_outputs = {
        section: {
            key: _relative_display_path(path, base_dir=out_dir)
            for key, path in values.items()
        }
        for section, values in combined_outputs.items()
    }
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
        f"- GT positive threshold: {describe_gt_rule(fdr_threshold, abs_logfc_threshold, markdown=True)}",
        (
            f"- Predictor agreement top fraction: `{predictor_top_fraction:.0%}`"
            " (exact top-k per predictor, deterministic tie-break)"
        ),
        "- Score handling: predictors are first aligned so that higher always means stronger",
        "- Per-dataset heatmaps and agreement plots: dataset-local tie-aware dense ranking over scored rows",
        "- Cross-dataset rank-distribution plots: global tie-aware dense ranking over each predictor's full standardized file",
        "- Combined PR/ROC/GSEA comparison plots: computed on the common set of genes scored by all compared predictors",
        "",
        "## Cross-Dataset Summary",
        (
            "Coverage, positive coverage, and mean metric performance are summarized numerically below. "
            "The PDF report carries the same summary table, so there is no separate combined coverage scatter "
            "or mean-metric heatmap in this run package."
        ),
        "",
    ]
    if summary_df is not None and not summary_df.empty:
        lines.extend(_cross_dataset_markdown_table(summary_df))
    else:
        lines.append("Cross-dataset summary table is unavailable for this run.")
    lines.extend(
        [
            "",
        "## Where To Start",
        "- Read `REPORT.pdf` for the main run-level summary with combined plots and explanatory notes",
        "- Open `tables/combined/cross_dataset_predictor_summary.tsv` for the compact numeric cross-dataset summary",
        "- Browse `plots/combined/` for cross-dataset comparison figures",
        "- Browse `datasets/<dataset_id>/` for dataset-specific tables, plots, and reports",
        "",
        "## Layout",
        "- `datasets/<dataset_id>/joined.tsv`: joined DE + predictor score table for one dataset",
        "- `datasets/<dataset_id>/plots/predictors/<tool_id>/`: per-tool visual outputs",
        "- `datasets/<dataset_id>/plots/comparisons/`: multi-predictor comparison plots for that dataset",
        "- `datasets/<dataset_id>/plots/heatmaps/`: dataset-level heatmaps",
        "- `datasets/<dataset_id>/reports/`: per-dataset Markdown/PDF reports and correlation TSVs",
        "- `tables/per_experiment/`: metric tables across datasets, one row per experiment",
        "- `tables/combined/`: cross-dataset predictor summary table",
        "- `plots/combined/metrics/`, `plots/combined/coverage/`, `plots/combined/ranks/`: cross-dataset comparison plots grouped by theme",
        "- `summary.json`: machine-readable run summary",
        "",
        "## Datasets",
        "| Dataset | miRNA | Perturbation | Cell line | Joined table |",
        "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in dataset_outputs:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['dataset_id']}`",
                    f"`{item['mirna']}`",
                    f"`{item['perturbation']}`",
                    f"`{item['cell_line']}`",
                    f"`{_relative_display_path(item['joined_tsv'], base_dir=out_dir)}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Key Files",
            "- `REPORT.pdf`: polished run-level summary with the main cross-dataset table and selected plots",
            "- `tables/combined/cross_dataset_predictor_summary.tsv`: exact numeric cross-dataset summary used in the report",
            "",
            "### Per-Experiment Tables",
        ]
    )
    metric_descriptions = {
        "coverage": "fraction of genes with predictor scores",
        "positive_coverage": "fraction of GT-positive genes that were scored",
        "aps": "average precision score per dataset",
        "pr_auc": "precision-recall AUC per dataset",
        "spearman": "score vs expected-effect Spearman correlation per dataset",
        "auroc": "AUROC per dataset",
    }
    for key, path in relative_metric_tables.items():
        lines.append(f"- `{path}`: {metric_descriptions.get(key, key)}")
    lines.extend(
        [
            "",
            "### Combined Plots",
        ]
    )
    combined_plot_descriptions = {
        "cross_dataset_metric_distributions": "distribution of each metric across the selected datasets",
        "positive_coverage_vs_performance": "mean positive coverage against mean APS and mean AUROC",
        "positive_background_rank_distributions": "global-rank separation of GT positives from background genes",
    }
    for key, path in relative_combined_outputs.get("plots", {}).items():
        lines.append(f"- `{path}`: {combined_plot_descriptions.get(key, key)}")
    if comparison_runs:
        lines.extend(
            [
                "",
                "### Comparison Variants",
            ]
        )
        for label, info in comparison_runs.items():
            variant_dir = info.get("out_dir")
            variant_readme = info.get("readme")
            parts = [f"`{label}`"]
            if variant_dir:
                parts.append(f"directory `{_relative_display_path(variant_dir, base_dir=out_dir)}`")
            if variant_readme:
                parts.append(f"README `{_relative_display_path(variant_readme, base_dir=out_dir)}`")
            lines.append("- " + " | ".join(parts))
    lines.extend(
        [
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
    summary_df = _load_cross_dataset_summary(combined_outputs)
    report_path = out_dir / "REPORT.pdf"

    with PdfPages(report_path) as pdf:
        def new_page():
            page_fig, page_ax = plt.subplots(figsize=(8.27, 11.69))
            page_ax.axis("off")
            page_fig.patch.set_facecolor("white")
            return page_fig, page_ax

        def add_header(ax, title, subtitle=None):
            ax.text(
                0.06,
                0.95,
                title,
                fontsize=20,
                fontweight="bold",
                color="#17324D",
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
            if subtitle:
                ax.text(
                    0.06,
                    0.915,
                    subtitle,
                    fontsize=10.5,
                    color="#5B6577",
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                )
            ax.add_line(
                plt.Line2D([0.06, 0.94], [0.895, 0.895], color="#D8DEE9", linewidth=1.4)
            )

        def add_block(ax, title, lines, *, x, y, width):
            ax.text(
                x,
                y,
                title,
                fontsize=11.5,
                fontweight="bold",
                color="#2F5D8C",
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
            current_y = y - 0.03
            for line in lines:
                wrapped = textwrap.wrap(line, width=max(26, int(width * 95))) or [""]
                for chunk in wrapped:
                    ax.text(
                        x,
                        current_y,
                        chunk,
                        fontsize=9.5,
                        color="#22303C",
                        va="top",
                        ha="left",
                        family="DejaVu Sans",
                    )
                    current_y -= 0.024
                current_y -= 0.004
            return current_y

        fig, ax = new_page()
        add_header(
            ax,
            "FuNmiRBench Run Report",
            f"{len(dataset_outputs)} datasets | {len(tool_ids)} predictors | generated from {config_path.name}",
        )
        summary_boxes = [
            ("Datasets", str(len(dataset_outputs))),
            ("Predictors", str(len(tool_ids))),
            ("GT Threshold", _gt_threshold_box_text(fdr_threshold, abs_logfc_threshold)),
            ("Top Fraction", f"{predictor_top_fraction:.0%}\nexact top-k"),
        ]
        x_positions = [0.06, 0.29, 0.52, 0.75]
        for (label, value), x in zip(summary_boxes, x_positions):
            ax.text(
                x,
                0.84,
                f"{label}\n{value}",
                fontsize=10.5,
                fontweight="bold",
                color="#17324D",
                va="top",
                ha="left",
                family="DejaVu Sans",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "#F5F8FC",
                    "edgecolor": "#D8E2EF",
                },
            )
        add_block(
            ax,
            "Evaluation Settings",
            [
                f"GT positives: {describe_gt_rule(fdr_threshold, abs_logfc_threshold)}",
                "Predictor scores are aligned so that higher always means stronger before evaluation.",
                "Per-dataset heatmaps and agreement plots use dataset-local tie-aware dense ranks.",
                "Combined PR/ROC/GSEA plots use only the common set of genes scored by all compared predictors.",
            ],
            x=0.06,
            y=0.69,
            width=0.40,
        )
        add_block(
            ax,
            "What This Report Emphasizes",
            [
                "Cross-dataset coverage and performance are summarized numerically in the predictor table on the next page.",
                "The exact numeric source for the report is tables/combined/cross_dataset_predictor_summary.tsv.",
            ],
            x=0.54,
            y=0.69,
            width=0.38,
        )
        add_block(
            ax,
            "Key Files",
            [
                "REPORT.pdf: this run-level summary",
                "tables/combined/cross_dataset_predictor_summary.tsv: exact mean/median/std values by predictor",
                "tables/per_experiment/: per-dataset APS, PR-AUC, AUROC, coverage, positive coverage, and Spearman tables",
                "datasets/<dataset_id>/: joined tables, plots, and per-predictor reports",
            ],
            x=0.06,
            y=0.36,
            width=0.86,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = new_page()
        add_header(ax, "Cross-Dataset Summary", "Mean values are across the selected datasets only.")
        takeaways = _report_takeaways(summary_df)
        if takeaways:
            add_block(ax, "Quick Takeaways", takeaways, x=0.06, y=0.85, width=0.88)
        if summary_df is not None and not summary_df.empty:
            display_df = summary_df[
                [
                    "tool_id",
                    "coverage_mean",
                    "positive_coverage_mean",
                    "aps_mean",
                    "pr_auc_mean",
                    "spearman_mean",
                    "auroc_mean",
                ]
            ].copy()
            display_df.columns = [
                "Predictor",
                "Mean coverage",
                "Mean positive coverage",
                "Mean APS",
                "Mean PR-AUC",
                "Mean Spearman",
                "Mean AUROC",
            ]
            for column in display_df.columns[1:]:
                percent = "coverage" in column.lower()
                display_df[column] = display_df[column].map(
                    lambda value: _format_summary_value(value, percent=percent)
                )
            table = ax.table(
                cellText=display_df.values.tolist(),
                colLabels=display_df.columns.tolist(),
                cellLoc="center",
                colLoc="center",
                bbox=[0.06, 0.18, 0.88, 0.48],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.8)
            table.scale(1.0, 1.35)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#E9F1FB")
                    cell.set_edgecolor("#D8E2EF")
                    cell.set_text_props(weight="bold", color="#17324D")
                else:
                    cell.set_edgecolor("#E1E8F0")
                    cell.set_facecolor("#FFFFFF" if row % 2 else "#F9FBFD")
            ax.text(
                0.06,
                0.11,
                (
                    "Coverage columns replace the removed combined coverage scatter, and the mean metric columns replace "
                    "the removed combined metric heatmap. Use the TSV for the full count/median/std/min/max summary."
                ),
                fontsize=9.2,
                color="#22303C",
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = new_page()
        add_header(ax, "Dataset Inventory", "Datasets included in this benchmark run.")
        dataset_df = pd.DataFrame(
            [
                {
                    "Dataset": item["dataset_id"],
                    "miRNA": item["mirna"],
                    "Perturbation": item["perturbation"],
                    "Cell line": item["cell_line"],
                }
                for item in dataset_outputs
            ]
        )
        table = ax.table(
            cellText=dataset_df.values.tolist(),
            colLabels=dataset_df.columns.tolist(),
            cellLoc="left",
            colLoc="left",
            bbox=[0.06, 0.56, 0.88, 0.24],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.2)
        table.scale(1.0, 1.45)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#E9F1FB")
                cell.set_edgecolor("#D8E2EF")
                cell.set_text_props(weight="bold", color="#17324D")
            else:
                cell.set_edgecolor("#E1E8F0")
                cell.set_facecolor("#FFFFFF" if row % 2 else "#F9FBFD")
        add_block(
            ax,
            "Included Combined Figures",
            [
                "cross_dataset_metric_distributions.png: how each metric varies across the selected datasets",
                "positive_coverage_vs_performance.png: mean positive coverage against mean APS and AUROC",
                "positive_background_rank_distributions.png: whether positives rank above background on the global rank scale",
            ],
            x=0.06,
            y=0.45,
            width=0.88,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        plot_items = []
        plot_descriptions = {
            "cross_dataset_metric_distributions": (
                "Cross-dataset metric distributions",
                "Each panel shows the spread of one metric across datasets for every predictor. "
                "Spearman uses the full -1 to 1 range so weak negative correlations remain visible."
            ),
            "positive_coverage_vs_performance": (
                "Positive coverage vs performance",
                "Mean positive coverage is plotted against mean APS and mean AUROC. "
                "This is especially helpful for sparse predictors where overall coverage alone can be misleading."
            ),
            "positive_background_rank_distributions": (
                "Positive vs background rank distributions",
                "Global-rank distributions aggregated across datasets, split into GT positives and background genes. "
                "Stronger predictors should push positives higher than background."
            ),
        }
        for key, (title, caption) in plot_descriptions.items():
            path = combined_outputs.get("plots", {}).get(key)
            if path:
                plot_items.append((title, caption, pathlib.Path(path)))

        for title, caption, path in plot_items:
            if not path.is_file():
                continue
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            fig.patch.set_facecolor("white")
            ax.text(
                0.06,
                0.975,
                title,
                fontsize=12,
                fontweight="bold",
                color="#17324D",
                va="top",
                ha="left",
            )
            ax.text(
                0.06,
                0.94,
                caption,
                fontsize=9.4,
                color="#22303C",
                va="top",
                ha="left",
                wrap=True,
            )
            ax.imshow(image)
            ax.set_position([0.08, 0.08, 0.84, 0.78])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return report_path


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
    fdr_threshold,
    abs_logfc_threshold,
    predictor_top_fraction,
    comparison_runs=None,
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
        predictor_top_fraction=predictor_top_fraction,
        comparison_runs=comparison_runs,
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
        **({"comparison_runs": comparison_runs} if comparison_runs else {}),
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
    no_fdr_out_dir = out_dir / NO_FDR_COMPARISON_DIRNAME
    no_fdr_layout = _init_run_layout(no_fdr_out_dir)

    tool_ids = list(predictions)
    metric_rows = []
    dataset_outputs = []
    joined_frames = []
    no_fdr_metric_rows = []
    no_fdr_dataset_outputs = []
    no_fdr_joined_frames = []
    fdr_threshold = float(eval_cfg.get("fdr_threshold", 0.05))
    abs_logfc_threshold = float(eval_cfg.get("abs_logfc_threshold", 1.0))
    predictor_top_fraction = float(eval_cfg.get("predictor_top_fraction", 0.10))
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
        no_fdr_dataset_dir = no_fdr_layout["datasets_dir"] / meta.id
        for branch_dir in (dataset_dir, no_fdr_dataset_dir):
            if branch_dir.exists():
                shutil.rmtree(branch_dir)
            (branch_dir / "plots").mkdir(parents=True, exist_ok=True)
            (branch_dir / "reports").mkdir(parents=True, exist_ok=True)
        logger.info(f"  Joining predictions for {meta.id}...")
        joined, predictor_output_paths = build_joined(meta, tool_ids, predictions, root)
        joined_path = dataset_dir / "joined.tsv"
        joined.to_csv(joined_path, sep="\t", index=False)
        no_fdr_joined_path = no_fdr_dataset_dir / "joined.tsv"
        joined.to_csv(no_fdr_joined_path, sep="\t", index=False)
        logger.info(f"  Wrote joined table: {joined_path}")
        joined_frames.append(joined.copy())
        no_fdr_joined_frames.append(joined.copy())

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
            logger=logger.info,
        )
        logger.info(f"  Evaluating no-FDR comparison metrics and plots for {meta.id}...")
        no_fdr_evaluation = evaluate_joined_dataframe(
            joined,
            plots_dir=no_fdr_dataset_dir / "plots",
            reports_dir=no_fdr_dataset_dir / "reports",
            fdr_threshold=None,
            abs_logfc_threshold=abs_logfc_threshold,
            predictor_top_fraction=predictor_top_fraction,
            dataset_id=meta.id,
            mirna=meta.miRNA,
            cell_line=meta.cell_line,
            perturbation=meta.perturbation,
            geo_accession=meta.geo_accession,
            de_table_path=str(meta.full_path),
            joined_tsv=no_fdr_joined_path,
            predictor_output_paths=predictor_output_paths,
            logger=logger.info,
        )
        metric_rows.extend(evaluation["metric_rows"])
        no_fdr_metric_rows.extend(no_fdr_evaluation["metric_rows"])
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
        no_fdr_dataset_outputs.append(
            {
                "dataset_id": meta.id,
                "mirna": meta.miRNA,
                "cell_line": meta.cell_line,
                "perturbation": meta.perturbation,
                "geo_accession": meta.geo_accession,
                "de_table_path": str(meta.full_path),
                "joined_tsv": str(no_fdr_joined_path),
                "dataset_dir": str(no_fdr_dataset_dir),
                "predictor_output_paths": predictor_output_paths,
                "plots": no_fdr_evaluation["plots"],
                "predictor_correlation_tsv": no_fdr_evaluation["predictor_correlation_tsv"],
            }
        )
        logger.info(f"  Finished {meta.id}")

    logger.info("Writing metric tables...")
    logger.info("Writing cross-dataset summaries...")
    no_fdr_result = _finalize_run_bundle(
        no_fdr_out_dir,
        out_root=out_root,
        config_path=config_path,
        tags=config.get("tags"),
        dataset_outputs=no_fdr_dataset_outputs,
        tool_ids=tool_ids,
        metric_rows=no_fdr_metric_rows,
        joined_frames=no_fdr_joined_frames,
        fdr_threshold=None,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
    )
    comparison_runs = {
        NO_FDR_COMPARISON_DIRNAME: {
            "out_dir": str(no_fdr_out_dir),
            "readme": str(no_fdr_result["readme_path"]),
            "report_pdf": str(no_fdr_result["report_path"]),
            "summary": str(no_fdr_result["summary_path"]),
        }
    }
    main_result = _finalize_run_bundle(
        out_dir,
        out_root=out_root,
        config_path=config_path,
        tags=config.get("tags"),
        dataset_outputs=dataset_outputs,
        tool_ids=tool_ids,
        metric_rows=metric_rows,
        joined_frames=joined_frames,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
        comparison_runs=comparison_runs,
    )
    return out_dir


def main():
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))
    out_dir = run_benchmark(args.config)
    logger.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
