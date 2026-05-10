"""Publication-focused PDF report helpers.

The benchmark still writes machine-readable README/TSV/JSON outputs separately.
This module is only responsible for making the top-level PDF suitable as a
shareable publication-style product: stable spacing, readable tables, clear
captions, and no stale diagnostic wording.
"""

from __future__ import annotations

import pathlib
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from funmirbench.evaluate import REPORT_PAGE_SIZE, describe_gt_rule
from funmirbench.benchmark_reports import (
    MIN_HEADLINE_COVERAGE,
    _coverage_analysis_lines,
    _format_summary_value,
    _gt_threshold_box_text,
    _load_cross_dataset_summary,
    _report_takeaways,
)


PUBLICATION_BLUE = "#17324D"
PUBLICATION_MUTED = "#5B6577"
PUBLICATION_RULE = "#D8DEE9"
PUBLICATION_TABLE_HEADER = "#E9F1FB"
PUBLICATION_TABLE_ALT = "#F9FBFD"


def _new_page():
    fig = plt.figure(figsize=REPORT_PAGE_SIZE)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def _save_page(pdf, fig):
    fig.set_size_inches(*REPORT_PAGE_SIZE, forward=True)
    fig.patch.set_facecolor("white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _header(ax, title, subtitle=None):
    ax.text(
        0.06,
        0.955,
        title,
        fontsize=20,
        fontweight="bold",
        color=PUBLICATION_BLUE,
        va="top",
        ha="left",
        family="DejaVu Sans",
    )
    if subtitle:
        ax.text(
            0.06,
            0.918,
            subtitle,
            fontsize=10.2,
            color=PUBLICATION_MUTED,
            va="top",
            ha="left",
            family="DejaVu Sans",
        )
    ax.add_line(plt.Line2D([0.06, 0.94], [0.895, 0.895], color=PUBLICATION_RULE, linewidth=1.2))


def _text_block(ax, title, lines, *, x, y, width, body_size=9.1, title_size=11.2):
    ax.text(
        x,
        y,
        title,
        fontsize=title_size,
        fontweight="bold",
        color="#2F5D8C",
        va="top",
        ha="left",
        family="DejaVu Sans",
    )
    current_y = y - 0.032
    wrap_width = max(30, int(width * 108))
    for line in lines:
        for chunk in textwrap.wrap(str(line), width=wrap_width) or [""]:
            ax.text(
                x,
                current_y,
                chunk,
                fontsize=body_size,
                color="#22303C",
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
            current_y -= 0.022
        current_y -= 0.006
    return current_y


def _summary_box(ax, label, value, *, x, y, width=0.18):
    ax.text(
        x,
        y,
        f"{label}\n{value}",
        fontsize=10.1,
        fontweight="bold",
        color=PUBLICATION_BLUE,
        va="top",
        ha="left",
        family="DejaVu Sans",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#F5F8FC",
            "edgecolor": "#D8E2EF",
        },
    )


def _format_summary_table(summary_df):
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
        "Mean\ncoverage",
        "Mean positive\ncoverage",
        "Mean\nAPS",
        "Mean\nPR-AUC",
        "Mean\nSpearman",
        "Mean\nAUROC",
    ]
    for column in display_df.columns[1:]:
        percent = "coverage" in column.lower()
        display_df[column] = display_df[column].map(lambda value: _format_summary_value(value, percent=percent))
    return display_df


def _draw_summary_table(ax, summary_df, *, bbox):
    display_df = _format_summary_table(summary_df)
    col_widths = [0.19, 0.12, 0.17, 0.11, 0.12, 0.14, 0.12]
    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        colWidths=col_widths,
        cellLoc="center",
        colLoc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2EC")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(PUBLICATION_TABLE_HEADER)
            cell.set_text_props(weight="bold", color=PUBLICATION_BLUE, fontsize=6.8)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else PUBLICATION_TABLE_ALT)
            if col == 0:
                cell.set_text_props(ha="left")
    return table


def _plot_items(combined_outputs):
    descriptions = {}
    for metric_name in ["coverage", "positive_coverage", "aps", "pr_auc", "spearman", "auroc"]:
        descriptions[f"cross_dataset_{metric_name}_distribution"] = (
            f"Cross-dataset {metric_name.upper()} distribution",
            f"Distribution of {metric_name.upper()} across selected datasets for each predictor.",
        )
    descriptions.update(
        {
            "positive_background_local_rank_distributions": (
                "Positive vs background local rank distributions",
                "Dataset-local rank distributions aggregated across datasets; stronger predictors shift positives rightward.",
            ),
            "positive_background_global_rank_distributions": (
                "Positive vs background global rank distributions",
                "Predictor-global rank distributions aggregated across datasets, using each predictor's full standardized file.",
            ),
        }
    )
    for key, (title, caption) in descriptions.items():
        path = combined_outputs.get("plots", {}).get(key)
        if path:
            path = pathlib.Path(path)
            if path.is_file():
                yield title, caption, path


def write_publication_run_pdf_report(
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
    del metric_tables
    summary_df = _load_cross_dataset_summary(combined_outputs)
    report_path = pathlib.Path(out_dir) / "REPORT.pdf"

    with PdfPages(report_path) as pdf:
        fig, ax = _new_page()
        _header(
            ax,
            "FuNmiRBench Benchmark Report",
            f"{len(dataset_outputs)} datasets | {len(tool_ids)} predictors | generated from {pathlib.Path(config_path).name}",
        )
        for label, value, x in [
            ("Datasets", str(len(dataset_outputs)), 0.06),
            ("Predictors", str(len(tool_ids)), 0.29),
            ("GT threshold", _gt_threshold_box_text(fdr_threshold, abs_logfc_threshold), 0.52),
            ("Top fraction", f"{predictor_top_fraction:.0%}\nexact top-k", 0.75),
        ]:
            _summary_box(ax, label, value, x=x, y=0.842)

        left_y = _text_block(
            ax,
            "Evaluation design",
            [
                f"GT positives: {describe_gt_rule(fdr_threshold, abs_logfc_threshold)}.",
                "Predictor scores are aligned so that higher scores consistently indicate stronger predicted targeting.",
                "Dataset-level heatmaps and agreement summaries use dataset-local, tie-aware dense ranks.",
                "Cross-dataset rank plots use predictor-global ranks from each full standardized predictor file.",
            ],
            x=0.06,
            y=0.69,
            width=0.40,
        )
        right_y = _text_block(
            ax,
            "How to read this report",
            [
                "The cross-dataset table is the primary benchmark summary.",
                "Sparse predictors are shown but are not promoted into headline rankings unless they pass the coverage threshold.",
                "Top-prediction effect CDFs are written for each dataset when enabled in the benchmark configuration.",
            ],
            x=0.54,
            y=0.69,
            width=0.38,
        )
        _text_block(
            ax,
            "Primary files",
            [
                "tables/combined/cross_dataset_predictor_summary.tsv: exact numeric summary used for the report.",
                "tables/per_experiment/: per-dataset metric tables for coverage, APS, PR-AUC, Spearman, and AUROC.",
                "datasets/<dataset_id>/: joined tables, publication plots, and per-predictor reports.",
            ],
            x=0.06,
            y=min(left_y, right_y) - 0.035,
            width=0.86,
        )
        _save_page(pdf, fig)

        fig, ax = _new_page()
        _header(ax, "Cross-Dataset Predictor Summary", "Coverage-aware interpretation of mean performance across selected datasets.")
        current_y = 0.85
        if summary_df is not None and not summary_df.empty:
            takeaways = _report_takeaways(summary_df)
            if takeaways:
                current_y = _text_block(ax, "Key results", takeaways, x=0.06, y=current_y, width=0.88, body_size=8.7) - 0.015
            coverage_lines = _coverage_analysis_lines(summary_df)
            if coverage_lines:
                current_y = _text_block(ax, "Coverage interpretation", coverage_lines, x=0.06, y=current_y, width=0.88, body_size=8.7) - 0.02
            table_top = min(0.50, current_y)
            _draw_summary_table(ax, summary_df, bbox=[0.035, 0.18, 0.93, max(0.29, table_top - 0.18)])
            footer = (
                f"Headline non-oracle rankings require >= {MIN_HEADLINE_COVERAGE:.0%} mean coverage. "
                "This prevents very sparse predictors from dominating based on a small evaluated subset. "
                "Use the TSV for count, median, standard deviation, minimum, and maximum values."
            )
            for i, line in enumerate(textwrap.wrap(footer, width=118)):
                ax.text(0.06, 0.125 - i * 0.023, line, fontsize=8.5, color="#22303C", va="top", ha="left")
        else:
            _text_block(ax, "No summary table", ["Cross-dataset summary table is unavailable for this run."], x=0.06, y=0.83, width=0.88)
        _save_page(pdf, fig)

        fig, ax = _new_page()
        _header(ax, "Dataset Inventory", "Datasets included in this benchmark run.")
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
            bbox=[0.06, 0.57, 0.88, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.9)
        table.scale(1.0, 1.45)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D9E2EC")
            if row == 0:
                cell.set_facecolor(PUBLICATION_TABLE_HEADER)
                cell.set_text_props(weight="bold", color=PUBLICATION_BLUE)
            else:
                cell.set_facecolor("#FFFFFF" if row % 2 else PUBLICATION_TABLE_ALT)
        _text_block(
            ax,
            "Included figure families",
            [
                "Combined metric distributions summarize per-dataset metric spread by predictor.",
                "Rank-distribution plots compare GT-positive genes against background genes across datasets.",
                "Per-dataset directories contain CDFs, common-set comparisons, own-scored comparisons, heatmaps, and per-predictor reports.",
            ],
            x=0.06,
            y=0.46,
            width=0.88,
        )
        _save_page(pdf, fig)

        for title, caption, path in _plot_items(combined_outputs):
            image = plt.imread(path)
            fig, _ = _new_page()
            fig.text(0.06, 0.975, title, fontsize=12, fontweight="bold", color=PUBLICATION_BLUE, va="top", ha="left")
            for i, line in enumerate(textwrap.wrap(caption, width=112)):
                fig.text(0.06, 0.942 - i * 0.021, line, fontsize=9.1, color="#22303C", va="top", ha="left")
            image_ax = fig.add_axes([0.035, 0.035, 0.93, 0.83])
            image_ax.imshow(image)
            image_ax.axis("off")
            _save_page(pdf, fig)

    return report_path
