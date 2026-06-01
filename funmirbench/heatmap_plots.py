"""Readable heatmap plot helpers for per-dataset benchmark outputs."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from funmirbench.evaluate_common import *


MAX_ALL_GENE_LABELS = 55
MAX_TOP_POSITIVE_LABELS = 180


def _heatmap_label_positions(labels, *, max_labels):
    """Return tick positions and labels, thinning only when rows are too dense."""
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
    if row_count <= 40:
        return 7.5
    if row_count <= 80:
        return 6.5
    if row_count <= 140:
        return 5.6
    return 5.0


def _top_heatmap_height(row_count):
    # Keep each row readable while avoiding extreme image sizes for very large
    # positive sets. Above the label cap, labels are thinned deterministically.
    return max(6.0, min(36.0, 2.2 + 0.18 * float(row_count)))


def _overview_heatmap_height(row_count):
    return max(6.0, min(22.0, 2.0 + 0.055 * float(row_count)))


def _prepare_heatmap_work(joined, *, required_cols, fdr_threshold, abs_logfc_threshold, perturbation):
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
    work = _sort_heatmap_rows_by_logfc(work)

    rank_frame = pd.DataFrame(
        {
            tool_id: work[rank_col].astype(float)
            for tool_id, rank_col in zip(tool_ids, rank_cols)
        }
    )

    max_abs_logfc = _nice_symmetric_limit(work["logFC"].to_numpy(dtype=float), floor=1.0)
    figure_height = _overview_heatmap_height(len(work))
    figure_width = max(11.5, 6.2 + len(tool_ids) * 1.05)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.48, 0.55, max(3.0, len(tool_ids) * 1.25)]},
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.18, wspace=0.20)
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(length=0, labelsize=8)

    gt_image = axes[0].imshow(
        work["is_positive"].to_numpy().reshape(-1, 1),
        aspect="auto",
        cmap=GT_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    del gt_image
    axes[0].set_title("GT status", fontsize=11, fontweight="semibold")
    axes[0].set_xticks([])

    logfc_image = axes[1].imshow(
        work["logFC"].to_numpy().reshape(-1, 1),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs_logfc, vcenter=0.0, vmax=max_abs_logfc),
        interpolation="nearest",
    )
    axes[1].set_title("logFC", fontsize=11, fontweight="semibold")
    axes[1].set_xticks([])

    score_cmap = PREDICTOR_HEATMAP_CMAP.copy()
    score_cmap.set_bad(MISSING_COLOR)
    heat = axes[2].imshow(
        np.ma.masked_invalid(rank_frame.to_numpy(dtype=float)),
        aspect="auto",
        cmap=score_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title("Predictor scores", fontsize=11, fontweight="semibold")
    axes[2].set_xticks(range(len(tool_ids)))
    axes[2].set_xticklabels(
        [_wrap_axis_label(_tool_label(tool_id)) for tool_id in tool_ids],
        rotation=0,
        ha="center",
    )

    labels = work["gene_id"].astype(str).tolist()
    positions, shown_labels = _heatmap_label_positions(labels, max_labels=MAX_ALL_GENE_LABELS)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        shown_labels,
        fontsize=_gene_label_fontsize(len(positions)),
        fontfamily="DejaVu Sans Mono",
        ha="right",
    )
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    axes[0].set_ylabel("genes ranked by effect", fontsize=PLOT_AXIS_LABEL_SIZE, color="#3C4858")

    _add_figure_heading(
        fig,
        title="Gene-level benchmarking overview",
        subtitle=(
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} genes ordered by perturbation-aware effect"
            "  |  dark GT = benchmark positive  |  gray cells indicate missing predictor pairs"
        ),
        title_y=0.985,
        subtitle_y=0.955,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=logfc_image,
        anchor_ax=axes[1],
        label="observed logFC",
        ticks=[-max_abs_logfc, 0.0, max_abs_logfc],
        height=0.012,
        pad=0.045,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=heat,
        anchor_ax=axes[2],
        label="dataset-local rank percentile",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        height=0.012,
        pad=0.045,
    )
    _save_figure(fig, out_path)


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

    rank_frame = pd.DataFrame(
        {
            tool_id: work[rank_col].astype(float)
            for tool_id, rank_col in zip(tool_ids, rank_cols)
        }
    )
    max_abs_logfc = _nice_symmetric_limit(work["logFC"].to_numpy(dtype=float), floor=1.0)
    figure_height = _top_heatmap_height(len(work))
    figure_width = max(12.0, 6.8 + len(tool_ids) * 1.10)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.80, max(3.4, len(tool_ids) * 1.25)]},
    )
    fig.subplots_adjust(left=0.30, right=0.985, top=0.92, bottom=0.13, wspace=0.18)
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
    positions, shown_labels = _heatmap_label_positions(labels, max_labels=MAX_TOP_POSITIVE_LABELS)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        shown_labels,
        fontsize=_gene_label_fontsize(len(positions)),
        fontfamily="DejaVu Sans Mono",
        ha="right",
    )
    axes[1].set_yticks([])
    axes[0].set_ylabel("top positives ranked by effect", fontsize=PLOT_AXIS_LABEL_SIZE, color="#3C4858")

    _add_figure_heading(
        fig,
        title=f"Top {int(positive_fraction * 100)}% of benchmark positives",
        subtitle=(
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} positive genes {_selection_caption(fdr_threshold)}"
            "  |  rows ordered by perturbation-aware effect"
        ),
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
    return True
