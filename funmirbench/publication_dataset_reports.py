"""Publication-quality per-predictor PDF reports."""

from __future__ import annotations

import pathlib
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from funmirbench.evaluate import REPORT_PAGE_SIZE, describe_gt_rule


BLUE = "#17324D"
MUTED = "#5B6577"
RULE = "#D8DEE9"
BOX_FACE = "#F5F8FC"
BOX_EDGE = "#D8E2EF"


def _metric_value(value, *, percent=False):
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    return f"{value:.1%}" if percent else f"{value:.3f}"


def _safe_relpath(path):
    if path is None:
        return "NA"
    return str(path).replace("\\", "/")


def _new_page():
    fig = plt.figure(figsize=REPORT_PAGE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def _save_page(pdf, fig):
    fig.set_size_inches(*REPORT_PAGE_SIZE, forward=True)
    fig.patch.set_facecolor("white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _header(ax, title, subtitle):
    ax.text(0.06, 0.95, title, fontsize=19, fontweight="bold", color=BLUE, va="top", ha="left")
    ax.text(0.06, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top", ha="left")
    ax.add_line(plt.Line2D([0.06, 0.94], [0.892, 0.892], color=RULE, linewidth=1.2))


def _metric_card(ax, label, value, *, x, y):
    ax.text(
        x,
        y,
        f"{label}\n{value}",
        fontsize=10.3,
        fontweight="bold",
        color=BLUE,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.42", "facecolor": BOX_FACE, "edgecolor": BOX_EDGE},
    )


def _block(ax, title, lines, *, x, y, width, body_size=9.2):
    ax.text(x, y, title, fontsize=11.3, fontweight="bold", color="#2F5D8C", va="top", ha="left")
    current_y = y - 0.034
    wrap_width = max(28, int(width * 105))
    for line in lines:
        for chunk in textwrap.wrap(str(line), width=wrap_width) or [""]:
            ax.text(x, current_y, chunk, fontsize=body_size, color="#22303C", va="top", ha="left")
            current_y -= 0.022
        current_y -= 0.008
    return current_y


def _plot_grid_page(pdf, *, title, subtitle, plot_paths):
    fig, ax = _new_page()
    _header(ax, title, subtitle)
    boxes = [
        (0.055, 0.50, 0.42, 0.34),
        (0.525, 0.50, 0.42, 0.34),
        (0.055, 0.09, 0.42, 0.34),
        (0.525, 0.09, 0.42, 0.34),
    ]
    for (label, path), box in zip(plot_paths, boxes):
        path = pathlib.Path(path)
        if not path.is_file():
            continue
        image = plt.imread(path)
        image_ax = fig.add_axes(box)
        image_ax.imshow(image)
        image_ax.axis("off")
        fig.text(box[0], box[1] + box[3] + 0.01, label, fontsize=8.8, color=MUTED, va="bottom", ha="left")
    _save_page(pdf, fig)


def write_publication_predictor_reports(
    *,
    reports_dir,
    plots_dir,
    dataset_id,
    mirna,
    cell_line,
    perturbation,
    geo_accession,
    de_table_path,
    predictor_output_paths,
    metric_rows,
    tool_labels,
    fdr_threshold,
    abs_logfc_threshold,
):
    reports_dir = pathlib.Path(reports_dir)
    plots_dir = pathlib.Path(plots_dir)
    written = []
    for row in metric_rows:
        tool_id = str(row.get("tool_id"))
        label = str(tool_labels.get(tool_id, tool_id))
        report_path = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.pdf"
        predictor_dir = plots_dir / "predictors" / tool_id
        plot_paths = [
            ("Score vs expected effect", predictor_dir / "score_vs_expected_effect.png"),
            ("GSEA enrichment", predictor_dir / "gsea_enrichment.png"),
            ("Precision-recall", predictor_dir / "precision_recall_curve.png"),
            ("ROC", predictor_dir / "roc_curve.png"),
        ]

        with PdfPages(report_path) as pdf:
            fig, ax = _new_page()
            _header(ax, f"{dataset_id} | {label}", f"{mirna} | {perturbation} | {cell_line}")
            cards = [
                ("Coverage", _metric_value(row.get("coverage"), percent=True), 0.06),
                ("Positive cov.", _metric_value(row.get("positive_coverage"), percent=True), 0.29),
                ("APS", _metric_value(row.get("aps")), 0.52),
                ("AUROC", _metric_value(row.get("auroc")), 0.75),
            ]
            for label_text, value, x in cards:
                _metric_card(ax, label_text, value, x=x, y=0.84)

            _block(
                ax,
                "Evaluation rule",
                [
                    f"GT positives: {describe_gt_rule(fdr_threshold, abs_logfc_threshold)}.",
                    "Scores are aligned so that higher values always indicate stronger predicted targeting before evaluation.",
                    "Metrics are computed on rows with usable ground truth and an available score for this predictor.",
                ],
                x=0.06,
                y=0.66,
                width=0.40,
            )
            _block(
                ax,
                "Coverage details",
                [
                    f"Rows total: {int(row.get('rows_total', 0)):,}",
                    f"Rows scored: {int(row.get('rows_scored', 0)):,}",
                    f"Rows missing score: {int(row.get('rows_missing_score', 0)):,}",
                    f"GT positives total: {int(row.get('positives_total', 0)):,}",
                    f"GT positives scored: {int(row.get('positives_scored', 0)):,}",
                ],
                x=0.54,
                y=0.66,
                width=0.38,
            )
            _block(
                ax,
                "Metric details",
                [
                    f"Pearson: {_metric_value(row.get('pearson'))}",
                    f"Spearman: {_metric_value(row.get('spearman'))}",
                    f"APS: {_metric_value(row.get('aps'))}",
                    f"PR-AUC: {_metric_value(row.get('pr_auc'))}",
                    f"AUROC: {_metric_value(row.get('auroc'))}",
                    f"GSEA ES: {_metric_value(row.get('gsea_es'))}",
                ],
                x=0.06,
                y=0.34,
                width=0.40,
            )
            _block(
                ax,
                "Provenance",
                [
                    f"GEO accession: {geo_accession or 'NA'}",
                    f"DE table: {_safe_relpath(de_table_path)}",
                    f"Predictor source: {_safe_relpath(predictor_output_paths.get(tool_id))}",
                ],
                x=0.54,
                y=0.34,
                width=0.38,
                body_size=8.6,
            )
            _save_page(pdf, fig)
            _plot_grid_page(
                pdf,
                title=f"{dataset_id} | {label} figures",
                subtitle="Per-predictor diagnostics used to interpret score ranking, enrichment, and classification behavior.",
                plot_paths=plot_paths,
            )
        written.append(report_path)
    return written
