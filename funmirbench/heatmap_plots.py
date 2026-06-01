"""Readable heatmap plot helpers for per-dataset benchmark outputs.

These helpers keep the benchmark row selection unchanged and only alter how
large heatmaps are rendered. Full overview heatmaps are still written to the
standard output paths; additional paginated files make gene IDs readable when
there are too many rows for one figure.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from funmirbench.evaluate_common import *


__all__ = [
    "_plot_algorithms_vs_genes_heatmap",
    "_plot_top_positive_heatmap",
]

MAX_OVERVIEW_LABELS = 45
TOP_POSITIVE_ROWS_PER_PAGE = 45


def _heatmap_label_positions(labels, *, max_labels):
    labels = list(labels)
    if not labels:
        return [], []
    if len(labels) <= max_labels:
        return list(range(len(labels))), labels
    step = int(math.ceil(len(labels) / float(max_labels)))
    positions = list(range(0, len(labels), step))
    if positions[-1] != len(labels) - 1:
        positions.append(len(labels) - 1)
    return positions, [labels[index] for index in positions]


def _gene_label_fontsize(row_count):
    if row_count <= 25:
        return 8.0
    if row_count <= 45:
        return 7.0
    return 6.0


def _overview_heatmap_height(row_count):
    return max(5.8, min(12.0, 2.0 + 0.035 * float(row_count)))


def _page_heatmap_height(row_count):
    return max(6.0, 2.0 + 0.22 * float(row_count))


def _prepare_heatmap_work(joined, *, required_cols, fdr_threshold, abs_logfc_threshold, perturbation):
    """Prepare the same usable rows and positives used by evaluation.

    Methodology:
    - effect-only mode keeps rows with usable logFC and ignores FDR.
    - FDR-aware mode keeps rows with usable logFC and valid 0 < FDR <= 1.
    - positives are then computed by _positive_mask(), shared with metrics.
    """
    work = joined[required_cols].copy()
    work = work[work["logFC"].notna()].copy()
    if fdr_threshold is not None:
        work["FDR"] = pd.to_numeric(work["FDR"], errors="coerce")
        work = work[work["FDR"].notna() & (work["FDR"] > 0.0) & (work["FDR"] <= 1.0)].copy()
    work = _annotate_ground_truth(work, perturbation=perturbation)
    work["is_positive"] = _positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
    return work


def _rank_frame(work, *, rank_cols, tool_ids):
    return pd.DataFrame(
        {
            tool_id: work[rank_col].astype(float)
            for tool_id, rank_col in zip(tool_ids, rank_cols)
        }
    )


def _draw_heatmap(
    work,
    *,
    rank_cols,
    tool_ids,
    dataset_id,
    out_path,
    title,
    subtitle,
    ylabel,
    show_all_labels,
    overview,
):
    if work.empty:
        return
    rank_frame = _rank_frame(work, rank_cols=rank_cols, tool_ids=tool_ids)
    max_abs_logfc = _nice_symmetric_limit(work["logFC"].to_numpy(dtype=float), floor=1.0)
    figure_height = _overview_heatmap_height(len(work)) if overview else _page_heatmap_height(len(work))
    figure_width = max(12.0, 6.8 + len(tool_ids) * 1.10)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.80, max(3.4, len(tool_ids) * 1.25)]},
    )
    fig.subplots_adjust(left=0.28, right=0.985, top=0.91, bottom=0.14, wspace=0.18)
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(length=0, labelsize=8)

    logfc_image = axes[0].imshow(
        work["logFC"].to_numpy().reshape(-1, 1),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs_logfc, vcenter=0.0, vmax=max_abs_logfc),
        interpolation="nearest",
    )
    axes[0].set_title("logFC", fontsize=11, fontweight="semibold")
    axes[0].set_xticks([])

    score_cmap = PREDICTOR_HEATMAP_CMAP.copy()
    score_cmap.set_bad(MISSING_COLOR)
    heat = axes[1].imshow(
        np.ma.masked_invalid(rank_frame.to_numpy(dtype=float)),
        aspect="auto",
        cmap=score_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("Predictor scores", fontsize=11, fontweight="semibold")
    axes[1].set_xticks(range(len(tool_ids)))
    axes[1].set_xticklabels(
        [_wrap_axis_label(_tool_label(tool_id)) for tool_id in tool_ids],
        rotation=0,
        ha="center",
    )

    labels = work["gene_id"].astype(str).tolist()
    if show_all_labels:
        positions, shown_labels = list(range(len(labels))), labels
    else:
        positions, shown_labels = _heatmap_label_positions(labels, max_labels=MAX_OVERVIEW_LABELS)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        shown_labels,
        fontsize=_gene_label_fontsize(len(positions)),
        fontfamily="DejaVu Sans Mono",
        ha="right",
    )
    axes[1].set_yticks([])
    axes[0].set_ylabel(ylabel, fontsize=PLOT_AXIS_LABEL_SIZE, color="#3C4858")

    _add_figure_heading(
        fig,
        title=title,
        subtitle=subtitle,
        title_y=0.985,
        subtitle_y=0.955,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=logfc_image,
        anchor_ax=axes[0],
        label="observed logFC",
        ticks=[-max_abs_logfc, 0.0, max_abs_logfc],
        height=0.012,
        pad=0.040,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=heat,
        anchor_ax=axes[1],
        label="dataset-local rank percentile",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        height=0.012,
        pad=0.040,
    )
    _save_figure(fig, out_path)


def _write_top_positive_label_pages(
    work,
    *,
    rank_cols,
    tool_ids,
    dataset_id,
    out_path,
    fdr_threshold,
):
    if len(work) <= TOP_POSITIVE_ROWS_PER_PAGE:
        return
    stem = out_path.with_suffix("")
    page_count = int(math.ceil(len(work) / float(TOP_POSITIVE_ROWS_PER_PAGE)))
    for page_index, start in enumerate(range(0, len(work), TOP_POSITIVE_ROWS_PER_PAGE), start=1):
        page = work.iloc[start : start + TOP_POSITIVE_ROWS_PER_PAGE].copy()
        page_path = stem.parent / f"{stem.name}_labels_page_{page_index:02d}.png"
        _draw_heatmap(
            page,
            rank_cols=rank_cols,
            tool_ids=tool_ids,
            dataset_id=dataset_id,
            out_path=page_path,
            title=f"Top benchmark positives, labels page {page_index}/{page_count}",
            subtitle=(
                f"{_dataset_caption(dataset_id)}  |  rows {start + 1}-{start + len(page)} of {len(work):,} "
                f"positive genes {_selection_caption(fdr_threshold)}"
            ),
            ylabel="gene_id",
            show_all_labels=True,
            overview=False,
        )


def _plot_algorithms_vs_genes_heatmap(
    joined, *, score_cols, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, perturbation=None,
):
    work = _prepare_heatmap_work(
        joined,
        required_cols=["gene_id", "logFC", "FDR", *score_cols, *rank_cols],
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        perturbation=perturbation,
    )
    if work.empty:
        return
    work = _sort_heatmap_rows_by_logfc(work)
    _draw_heatmap(
        work,
        rank_cols=rank_cols,
        tool_ids=tool_ids,
        dataset_id=dataset_id,
        out_path=out_path,
        title="Gene-level benchmarking overview",
        subtitle=(
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} genes ordered by perturbation-aware effect"
            "  |  gray cells indicate missing predictor pairs"
        ),
        ylabel="genes ranked by effect",
        show_all_labels=False,
        overview=True,
    )


def _plot_top_positive_heatmap(
    joined, *, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, positive_fraction, perturbation=None,
):
    work = _prepare_heatmap_work(
        joined,
        required_cols=["gene_id", "logFC", "FDR", *rank_cols],
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        perturbation=perturbation,
    )
    work = work[work["is_positive"] == 1].copy()
    if work.empty:
        return False

    work = _sort_heatmap_rows_by_logfc(work)
    rows_to_keep = max(1, int(math.ceil(len(work) * positive_fraction)))
    work = work.head(rows_to_keep).copy()

    _draw_heatmap(
        work,
        rank_cols=rank_cols,
        tool_ids=tool_ids,
        dataset_id=dataset_id,
        out_path=out_path,
        title=f"Top {int(positive_fraction * 100)}% of benchmark positives",
        subtitle=(
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} positive genes {_selection_caption(fdr_threshold)}"
            "  |  overview; labeled pages are written beside this file"
        ),
        ylabel="top positives ranked by effect",
        show_all_labels=False,
        overview=True,
    )
    _write_top_positive_label_pages(
        work,
        rank_cols=rank_cols,
        tool_ids=tool_ids,
        dataset_id=dataset_id,
        out_path=out_path,
        fdr_threshold=fdr_threshold,
    )
    return True
