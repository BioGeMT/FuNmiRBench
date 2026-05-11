"""Report rendering helpers for per-tool evaluation reports."""

import os
import pathlib
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from funmirbench.evaluate_common import *


def _relative_report_path(path, *, report_dir):
    if not path:
        return "NA"
    target = pathlib.Path(path).expanduser().resolve()
    base = pathlib.Path(report_dir).resolve()
    try:
        return os.path.relpath(target, start=base)
    except ValueError:
        return str(target)


def _build_tool_report_markdown(
    *, dataset_id, mirna, cell_line, perturbation, geo_accession,
    de_table_path, joined_tsv,
    tool_id, predictor_output_path, metrics, coverage_info,
    scatter_png, pr_curve_png, roc_curve_png, gsea_png,
    fdr_threshold, abs_logfc_threshold,
):
    coverage_percent = coverage_info["coverage"] * 100.0
    positive_coverage_percent = coverage_info["positive_coverage"] * 100.0
    lines = [
        f"# Evaluation Report: {dataset_id} | {_tool_label(tool_id)}",
        "",
        "## Snapshot",
        f"- predictor: `{tool_id}`",
        f"- dataset_id: `{dataset_id}`",
        f"- mirna: `{mirna or 'NA'}`",
        f"- cell_line: `{cell_line or 'NA'}`",
        f"- perturbation: `{perturbation or 'NA'}`",
        f"- geo_accession: `{geo_accession or 'NA'}`",
        (
            f"- overall coverage: `{coverage_percent:.1f}%`"
            f" (`{int(coverage_info['rows_scored'])}` of `{int(coverage_info['rows_total'])}` genes scored)"
        ),
        (
            f"- positive coverage: `{positive_coverage_percent:.1f}%`"
            f" (`{int(coverage_info['positives_scored'])}` of `{int(coverage_info['positives_total'])}` GT positives scored)"
        ),
        f"- aps: `{metrics['aps']:.6f}`",
        f"- auroc: `{metrics['auroc']:.6f}`",
        f"- spearman: `{metrics['spearman']:.6f}`",
        "",
        "## Evaluation Rule",
        f"- GT positives: {describe_gt_rule(fdr_threshold, abs_logfc_threshold, markdown=True)}",
        "- Predictor scores are aligned so that higher always means stronger before evaluation",
        "- Pearson and Spearman compare predictor score against perturbation-aware expected effect",
        "- APS, PR-AUC, AUROC, and GSEA are computed on scored rows only",
        "",
        "## Inputs",
        f"- de_table_path: `{de_table_path or 'NA'}`",
        f"- joined_tsv: `{joined_tsv or 'NA'}`",
        f"- tool_id: `{tool_id}`",
        f"- predictor_output_path: `{predictor_output_path or 'NA'}`",
        "",
        "## Coverage Details",
        f"- rows_total: `{int(coverage_info['rows_total'])}`",
        f"- rows_scored: `{int(coverage_info['rows_scored'])}`",
        f"- rows_missing_score: `{int(coverage_info['rows_missing_score'])}`",
        f"- coverage: `{coverage_info['coverage']:.6f}`",
        f"- positives_total: `{int(coverage_info['positives_total'])}`",
        f"- positives_scored: `{int(coverage_info['positives_scored'])}`",
        f"- positive_coverage: `{coverage_info['positive_coverage']:.6f}`",
        "",
        "## Metric Details",
        f"- rows_used: `{int(metrics['rows_used'])}`",
        f"- positives: `{int(metrics['positives'])}`",
        f"- negatives: `{int(metrics['negatives'])}`",
        f"- pearson: `{metrics['pearson']:.6f}`",
        f"- spearman: `{metrics['spearman']:.6f}`",
        f"- aps: `{metrics['aps']:.6f}`",
        f"- pr_auc: `{metrics['pr_auc']:.6f}`",
        f"- auroc: `{metrics['auroc']:.6f}`",
        "",
        "## Included Plots",
        f"- score_vs_expected_effect: `{scatter_png}`",
        f"- precision_recall_curve: `{pr_curve_png}`",
        f"- roc_curve: `{roc_curve_png}`",
        f"- gsea_enrichment: `{gsea_png}`",
        "",
    ]
    return "\n".join(lines)


def _render_tool_report_pdf(
    *,
    pdf_path,
    dataset_id,
    tool_id,
    mirna,
    cell_line,
    perturbation,
    geo_accession,
    predictor_output_path,
    metrics,
    coverage_info,
    fdr_threshold,
    abs_logfc_threshold,
    scatter_png=None,
    pr_curve_png=None,
    roc_curve_png=None,
    gsea_png=None,
):
    with PdfPages(pdf_path) as pdf:
        def new_page():
            page_fig = plt.figure(figsize=REPORT_PAGE_SIZE)
            page_ax = page_fig.add_axes([0.0, 0.0, 1.0, 1.0])
            page_ax.axis("off")
            page_fig.patch.set_facecolor("white")
            return page_fig, page_ax

        def add_header(ax, title, subtitle=None):
            ax.text(
                0.06,
                0.95,
                title,
                fontsize=19,
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
                    fontsize=10.3,
                    color="#5B6577",
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                )
            ax.add_line(plt.Line2D([0.06, 0.94], [0.892, 0.892], color="#D8DEE9", linewidth=1.4))

        def add_block(ax, title, lines, *, x, y, width):
            ax.text(
                x,
                y,
                title,
                fontsize=11.3,
                fontweight="bold",
                color="#2F5D8C",
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
            current_y = y - 0.03
            for line in lines:
                wrapped = textwrap.wrap(line, width=max(24, int(width * 94))) or [""]
                for chunk in wrapped:
                    ax.text(
                        x,
                        current_y,
                        chunk,
                        fontsize=9.4,
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
            f"{dataset_id} | {_tool_label(tool_id)}",
            f"{mirna or 'NA'} | {perturbation or 'NA'} | {cell_line or 'NA'}",
        )
        summary_cards = [
            ("Coverage", f"{coverage_info['coverage']:.1%}"),
            ("Positive cov", f"{coverage_info['positive_coverage']:.1%}"),
            ("APS", f"{metrics['aps']:.3f}"),
            ("AUROC", f"{metrics['auroc']:.3f}"),
        ]
        for (label, value), x in zip(summary_cards, [0.06, 0.29, 0.52, 0.75]):
            ax.text(
                x,
                0.84,
                f"{label}\n{value}",
                fontsize=10.4,
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
            "Evaluation Rule",
            [
                f"GT positives: {describe_gt_rule(fdr_threshold, abs_logfc_threshold)}",
                "Predictor scores are aligned so that higher always means stronger before evaluation.",
                "Pearson and Spearman compare score against perturbation-aware expected effect.",
                "APS, PR-AUC, AUROC, and GSEA are computed on scored rows only.",
            ],
            x=0.06,
            y=0.69,
            width=0.40,
        )
        add_block(
            ax,
            "Coverage Details",
            [
                f"rows_total: {int(coverage_info['rows_total'])}",
                f"rows_scored: {int(coverage_info['rows_scored'])}",
                f"rows_missing_score: {int(coverage_info['rows_missing_score'])}",
                f"positives_total: {int(coverage_info['positives_total'])}",
                f"positives_scored: {int(coverage_info['positives_scored'])}",
            ],
            x=0.54,
            y=0.69,
            width=0.38,
        )
        add_block(
            ax,
            "Metric Details",
            [
                f"Pearson: {metrics['pearson']:.3f}",
                f"Spearman: {metrics['spearman']:.3f}",
                f"APS: {metrics['aps']:.3f}",
                f"PR-AUC: {metrics['pr_auc']:.3f}",
                f"AUROC: {metrics['auroc']:.3f}",
            ],
            x=0.06,
            y=0.42,
            width=0.40,
        )
        add_block(
            ax,
            "Provenance",
            [
                f"GEO accession: {geo_accession or 'NA'}",
                f"Predictor source: {predictor_output_path or 'NA'}",
            ],
            x=0.54,
            y=0.42,
            width=0.38,
        )
        _save_pdf_page(pdf, fig)

        plot_specs = [
            ("Score vs expected effect", scatter_png),
            ("Precision-recall curve", pr_curve_png),
            ("ROC curve", roc_curve_png),
            ("GSEA enrichment", gsea_png),
        ]
        existing_plots = [(label, path) for label, path in plot_specs if path and pathlib.Path(path).is_file()]
        if existing_plots:
            fig, _ = new_page()
            fig.text(
                0.06,
                0.975,
                f"{dataset_id} | {_tool_label(tool_id)} | Plots",
                fontsize=13,
                fontweight="bold",
                color="#17324D",
                va="top",
                ha="left",
            )
            fig.text(
                0.06,
                0.945,
                "These are the main per-tool visuals for this dataset: ranking quality, classification quality, and enrichment behavior.",
                fontsize=9.2,
                color="#22303C",
                va="top",
                ha="left",
            )
            layout_specs = [
                ("Score vs expected effect", [0.08, 0.58, 0.84, 0.24]),
                ("Precision-recall curve", [0.06, 0.12, 0.27, 0.25]),
                ("ROC curve", [0.365, 0.12, 0.27, 0.25]),
                ("GSEA enrichment", [0.67, 0.12, 0.27, 0.25]),
            ]
            for label, bounds in layout_specs:
                match = next((path for plot_label, path in existing_plots if plot_label == label), None)
                if match is None:
                    continue
                image = plt.imread(match)
                fig.text(
                    bounds[0],
                    bounds[1] + bounds[3] + 0.015,
                    label,
                    fontsize=10,
                    fontweight="bold",
                    color="#2F5D8C",
                    ha="left",
                    va="bottom",
                )
                image_ax = fig.add_axes(bounds)
                image_ax.imshow(image)
                image_ax.axis("off")
            _save_pdf_page(pdf, fig)


def _write_tool_report(
    *, dataset_id, mirna, cell_line, perturbation, geo_accession,
    de_table_path, joined_tsv,
    tool_id, predictor_output_path, metrics, markdown_path, pdf_path, coverage_info,
    scatter_png, pr_curve_png, roc_curve_png, gsea_png, fdr_threshold, abs_logfc_threshold,
):
    report_dir = markdown_path.parent
    markdown_text = _build_tool_report_markdown(
        dataset_id=dataset_id,
        mirna=mirna,
        cell_line=cell_line,
        perturbation=perturbation,
        geo_accession=geo_accession,
        de_table_path=_relative_report_path(de_table_path, report_dir=report_dir),
        joined_tsv=_relative_report_path(joined_tsv, report_dir=report_dir),
        tool_id=tool_id,
        predictor_output_path=_relative_report_path(predictor_output_path, report_dir=report_dir),
        metrics=metrics,
        coverage_info=coverage_info,
        scatter_png=_relative_report_path(scatter_png, report_dir=report_dir),
        pr_curve_png=_relative_report_path(pr_curve_png, report_dir=report_dir),
        roc_curve_png=_relative_report_path(roc_curve_png, report_dir=report_dir),
        gsea_png=_relative_report_path(gsea_png, report_dir=report_dir),
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    )
    markdown_path.write_text(markdown_text + "\n", encoding="utf-8")
    _render_tool_report_pdf(
        pdf_path=pdf_path,
        dataset_id=dataset_id,
        tool_id=tool_id,
        mirna=mirna,
        cell_line=cell_line,
        perturbation=perturbation,
        geo_accession=geo_accession,
        predictor_output_path=_relative_report_path(predictor_output_path, report_dir=report_dir),
        metrics=metrics,
        coverage_info=coverage_info,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        scatter_png=scatter_png,
        pr_curve_png=pr_curve_png,
        roc_curve_png=roc_curve_png,
        gsea_png=gsea_png,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
