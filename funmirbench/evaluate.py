"""Evaluate joined GT/prediction tables: metrics, plots, reports."""

import math
import pathlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

SCORE_PREFIX = "score_"
GLOBAL_RANK_PREFIX = "global_rank_"
FIGURE_DPI = 300
NEGATIVE_COLOR = "#B8C4D6"
POSITIVE_COLOR = "#D04E4E"
NEUTRAL_COLOR = "#5B6577"
GRID_COLOR = "#D8DEE9"
SCORE_CMAP = ListedColormap(
    ["#F6F7FB", "#C5D7EE", "#7FA8D8", "#2F5D8C", "#17324D"]
)
GT_CMAP = ListedColormap(["#EEF1F6", "#243B53"])
MISSING_COLOR = "#EEF2F7"
CURVE_COLORS = [
    "#1F77B4",
    "#D1495B",
    "#2A9D8F",
    "#9467BD",
    "#D97D0D",
    "#4C78A8",
]


def _emit_log(logger, message):
    if logger is not None:
        logger(message)


def _tool_label(tool_id):
    return str(tool_id).replace("_", " ")


def _dataset_heading(dataset_id, *, suffix=None):
    if suffix:
        return f"{dataset_id} | {suffix}"
    return str(dataset_id)


def _dataset_caption(dataset_id):
    return str(dataset_id).replace("_", " ")


def _style_axes(ax, *, grid_axis="y"):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A0AEC0")
    ax.spines["bottom"].set_color("#A0AEC0")
    ax.tick_params(colors="#3C4858", labelsize=9)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _save_figure(fig, out_path):
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _safe_neglog10(series):
    clipped = series.astype(float).clip(lower=1e-300)
    return -clipped.map(math.log10)


def _rank_scale_scores(series):
    values = series.astype(float)
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(float("nan"), index=series.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=series.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def _tool_id_from_score_col(score_col):
    if score_col.startswith(SCORE_PREFIX):
        return score_col[len(SCORE_PREFIX):]
    return score_col


def _rank_col_for_tool(tool_id):
    return f"{GLOBAL_RANK_PREFIX}{tool_id}"


def _top_fraction_mask(series, fraction):
    valid = series.notna()
    if not bool(valid.any()):
        return pd.Series(False, index=series.index)
    threshold = series[valid].quantile(1.0 - fraction)
    return valid & (series >= threshold)


def _prepare_scored_frame(joined, *, score_col, fdr_threshold, abs_logfc_threshold):
    required_cols = {"gene_id", "logFC", "FDR", score_col}
    missing = [col for col in required_cols if col not in joined.columns]
    if missing:
        raise ValueError(f"Joined table missing required columns: {missing}")

    keep_cols = ["gene_id", "logFC", "FDR", score_col]
    for optional in ("dataset_id", "mirna", "PValue"):
        if optional in joined.columns:
            keep_cols.append(optional)

    keep = joined[keep_cols].copy()
    keep = keep[keep["logFC"].notna() & keep["FDR"].notna()].copy()
    keep = keep[keep["FDR"].astype(float) > 0].copy()
    if keep.empty:
        raise ValueError(f"No usable rows remain for {score_col}.")

    total_rows = int(len(keep))
    missing_score_count = int(keep[score_col].isna().sum())
    keep = keep[keep[score_col].notna()].copy()
    if keep.empty:
        raise ValueError(f"No scored rows remain for {score_col}.")

    rows_scored = int(len(keep))
    coverage = float(rows_scored / total_rows) if total_rows else float("nan")
    keep[score_col] = keep[score_col].astype(float)
    keep["logFC"] = keep["logFC"].astype(float)
    keep["FDR"] = keep["FDR"].astype(float)
    keep["abs_logFC"] = keep["logFC"].abs()
    keep["neglog10_FDR"] = _safe_neglog10(keep["FDR"])
    keep["is_positive"] = (
        (keep["FDR"] < fdr_threshold) & (keep["abs_logFC"] > abs_logfc_threshold)
    ).astype(int)
    positives = int(keep["is_positive"].sum())
    negatives = int(len(keep) - positives)
    if positives == 0:
        raise ValueError(f"No positives remain for {score_col}.")
    if negatives == 0:
        raise ValueError(f"No negatives remain for {score_col}.")
    coverage_info = {
        "rows_total": total_rows,
        "rows_scored": rows_scored,
        "rows_missing_score": missing_score_count,
        "coverage": coverage,
    }
    return keep, coverage_info


def _plot_scatter_with_correlation(df, *, score_col, dataset_id, tool_id, out_path):
    pearson = float(df[score_col].corr(df["logFC"], method="pearson"))
    spearman = float(df[score_col].corr(df["logFC"], method="spearman"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    positive_mask = df["is_positive"].astype(bool)
    negatives = df.loc[~positive_mask]
    positives = df.loc[positive_mask]

    _style_axes(ax, grid_axis="both")
    ax.scatter(
        negatives[score_col],
        negatives["logFC"],
        s=18,
        alpha=0.55,
        color=NEGATIVE_COLOR,
        edgecolors="none",
        label="background genes",
        rasterized=True,
    )
    ax.scatter(
        positives[score_col],
        positives["logFC"],
        s=28,
        alpha=0.9,
        color=POSITIVE_COLOR,
        edgecolors="white",
        linewidths=0.35,
        label="DE positives",
        rasterized=True,
        zorder=3,
    )
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle="--", alpha=0.75)
    score_min = float(df[score_col].min())
    score_max = float(df[score_col].max())
    if score_min <= 0.0 <= score_max:
        ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle=":", alpha=0.75)
    ax.set_xlabel(f"{_tool_label(tool_id)} score", fontsize=10)
    ax.set_ylabel("logFC", fontsize=10)
    ax.set_title(
        f"{_tool_label(tool_id)} score vs logFC",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(
        0.125,
        0.955,
        (
            f"{_dataset_caption(dataset_id)}"
            f"  |  n={len(df):,} genes"
            f"  |  positives={int(positive_mask.sum()):,}"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.text(
        0.99,
        0.02,
        f"Pearson {pearson:.3f}\nSpearman {spearman:.3f}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID_COLOR},
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _save_figure(fig, out_path)
    return pearson, spearman


def _plot_gsea_enrichment(df, *, score_col, dataset_id, tool_id, out_path):
    ordered = df.sort_values([score_col, "gene_id"], ascending=[False, True]).reset_index(drop=True)
    hits = ordered["is_positive"].astype(int).to_numpy(dtype=int)
    total_hits = int(hits.sum())
    total_misses = int(len(hits) - total_hits)
    if total_hits == 0 or total_misses == 0:
        raise ValueError(f"Cannot build enrichment plot for {tool_id}: need both hits and misses.")

    hit_step = 1.0 / total_hits
    miss_step = 1.0 / total_misses
    running_es = np.cumsum(np.where(hits == 1, hit_step, -miss_step))
    max_index = int(np.argmax(running_es))
    min_index = int(np.argmin(running_es))
    if abs(float(running_es[max_index])) >= abs(float(running_es[min_index])):
        es = float(running_es[max_index])
        peak_index = max_index
    else:
        es = float(running_es[min_index])
        peak_index = min_index

    positions = np.arange(1, len(ordered) + 1)
    hit_positions = positions[hits == 1]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.8, 1.6], "hspace": 0.08},
    )
    curve_ax, hit_ax, score_ax = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#3C4858", labelsize=9)

    curve_ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    curve_ax.set_axisbelow(True)
    curve_ax.plot(positions, running_es, color="#2F5D8C", linewidth=2.2)
    curve_ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle="--", alpha=0.8)
    curve_ax.scatter(
        [positions[peak_index]],
        [running_es[peak_index]],
        color=POSITIVE_COLOR if es >= 0 else "#4C78A8",
        s=26,
        zorder=3,
    )
    curve_ax.set_ylabel("Running ES", fontsize=10)
    curve_ax.set_title(
        f"{_tool_label(tool_id)} enrichment of GT positives",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(
        0.125,
        0.965,
        (
            f"{_dataset_caption(dataset_id)}"
            f"  |  positives={total_hits:,}/{len(ordered):,}"
            f"  |  ES={es:.3f}"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )

    hit_ax.vlines(hit_positions, 0.0, 1.0, color=POSITIVE_COLOR, linewidth=0.8)
    hit_ax.set_ylim(0.0, 1.0)
    hit_ax.set_yticks([])
    hit_ax.set_ylabel("Hits", fontsize=9)
    hit_ax.spines["left"].set_visible(False)
    hit_ax.spines["bottom"].set_visible(False)

    ordered_scores = ordered[score_col].to_numpy(dtype=float)
    score_ax.fill_between(positions, ordered_scores, 0.0, color="#BFD3EA", alpha=0.95)
    score_ax.plot(positions, ordered_scores, color="#577590", linewidth=1.2)
    score_ax.set_ylabel("Score", fontsize=9)
    score_ax.set_xlabel("Ranked genes", fontsize=10)
    score_ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    score_ax.set_axisbelow(True)

    _save_figure(fig, out_path)
    return es


def _compute_pr_metrics(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    aps = average_precision_score(y_true, y_score)
    return float(pr_auc), float(aps)


def _compute_auroc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))


def _plot_predictor_pr_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    _style_axes(ax, grid_axis="both")
    for index, item in enumerate(comparisons):
        precision, recall, _ = precision_recall_curve(item["y_true"], item["y_score"])
        pr_auc = auc(recall, precision)
        ax.plot(
            recall,
            precision,
            label=(
                f"{_tool_label(item['tool_id'])} "
                f"({pr_auc:.3f}, cov {item['coverage']:.0%})"
            ),
            linewidth=2.2,
            color=CURVE_COLORS[index % len(CURVE_COLORS)],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(
        "Precision-recall comparison",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(0.125, 0.955, _dataset_caption(dataset_id), fontsize=9, color=NEUTRAL_COLOR)
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_predictor_roc_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    _style_axes(ax, grid_axis="both")
    for index, item in enumerate(comparisons):
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_score"])
        auroc = roc_auc_score(item["y_true"], item["y_score"])
        ax.plot(
            fpr,
            tpr,
            label=(
                f"{_tool_label(item['tool_id'])} "
                f"({auroc:.3f}, cov {item['coverage']:.0%})"
            ),
            linewidth=2.2,
            color=CURVE_COLORS[index % len(CURVE_COLORS)],
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.4,
        color=NEUTRAL_COLOR,
        label="random",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.set_title(
        "ROC comparison",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(0.125, 0.955, _dataset_caption(dataset_id), fontsize=9, color=NEUTRAL_COLOR)
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_algorithms_vs_genes_heatmap(
    joined, *, score_cols, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold,
):
    work = joined[["gene_id", "logFC", "FDR", *score_cols, *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work["logFC"] = work["logFC"].astype(float)
    work["FDR"] = work["FDR"].astype(float)
    work["abs_logFC"] = work["logFC"].abs()
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold) & (work["abs_logFC"] > abs_logfc_threshold)
    ).astype(int)
    work = work.sort_values(
        ["is_positive", "FDR", "abs_logFC"], ascending=[False, True, False],
    ).reset_index(drop=True)

    rank_frame = pd.DataFrame(
        {
            tool_id: work[rank_col].astype(float)
            for tool_id, rank_col in zip(tool_ids, rank_cols)
        }
    )

    max_abs_logfc = max(float(work["abs_logFC"].max()), 1.0)
    figure_height = max(5.6, min(12, 0.025 * len(work)))
    figure_width = max(10.5, 5.2 + len(tool_ids) * 1.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.5, 0.6, max(2.8, len(tool_ids) * 1.25)]},
    )
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(length=0, labelsize=8)

    gt_image = axes[0].imshow(
        work["is_positive"].to_numpy().reshape(-1, 1), aspect="auto", cmap=GT_CMAP, vmin=0, vmax=1
    )
    axes[0].set_title("GT", fontsize=10, fontweight="semibold")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["positive"], rotation=90)

    logfc_image = axes[1].imshow(
        work["logFC"].to_numpy().reshape(-1, 1), aspect="auto", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs_logfc, vcenter=0.0, vmax=max_abs_logfc),
    )
    axes[1].set_title("logFC", fontsize=10, fontweight="semibold")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["logFC"], rotation=90)

    score_cmap = SCORE_CMAP.with_extremes(bad=MISSING_COLOR)
    heat = axes[2].imshow(
        np.ma.masked_invalid(rank_frame.to_numpy(dtype=float)),
        aspect="auto",
        cmap=score_cmap,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Predictor scores", fontsize=10, fontweight="semibold")
    axes[2].set_xticks(range(len(tool_ids)))
    axes[2].set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")

    if len(work) <= 40:
        labels = work["gene_id"].tolist()
        for axis in axes:
            axis.set_yticks(range(len(labels)))
            axis.set_yticklabels(labels, fontsize=7)
    else:
        for axis in axes:
            axis.set_yticks([])

    axes[0].set_ylabel("genes", fontsize=10, color="#3C4858")
    fig.suptitle(
        "Gene-level benchmarking overview",
        x=0.08,
        y=0.995,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.08,
        0.972,
        (
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} genes ordered by benchmark ground truth and effect size"
            "  |  blank cells indicate missing predictor pairs"
        ),
        fontsize=8.6,
        color=NEUTRAL_COLOR,
    )
    fig.colorbar(
        logfc_image,
        ax=axes[1],
        orientation="horizontal",
        fraction=0.08,
        pad=0.08,
        label="logFC",
    )
    fig.colorbar(
        heat,
        ax=axes[2],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        label="global rank percentile",
    )
    _save_figure(fig, out_path)


def _plot_top_positive_heatmap(
    joined, *, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, positive_fraction,
):
    work = joined[["gene_id", "logFC", "FDR", *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work["logFC"] = work["logFC"].astype(float)
    work["FDR"] = work["FDR"].astype(float)
    work["abs_logFC"] = work["logFC"].abs()
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold) & (work["abs_logFC"] > abs_logfc_threshold)
    ).astype(int)
    work = work[work["is_positive"] == 1].copy()
    if work.empty:
        return False

    work = work.sort_values(["FDR", "abs_logFC"], ascending=[True, False]).reset_index(drop=True)
    rows_to_keep = max(1, int(math.ceil(len(work) * positive_fraction)))
    work = work.head(rows_to_keep).copy()

    rank_frame = pd.DataFrame(
        {
            tool_id: work[rank_col].astype(float)
            for tool_id, rank_col in zip(tool_ids, rank_cols)
        }
    )
    max_abs_logfc = max(float(work["abs_logFC"].max()), 1.0)
    figure_height = max(4.2, min(9.0, 0.22 * len(work)))
    figure_width = max(10.5, 5.2 + len(tool_ids) * 1.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.7, max(2.8, len(tool_ids) * 1.25)]},
    )
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
    )
    axes[0].set_title("logFC", fontsize=10, fontweight="semibold")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["logFC"], rotation=90)

    score_cmap = SCORE_CMAP.with_extremes(bad=MISSING_COLOR)
    heat = axes[1].imshow(
        np.ma.masked_invalid(rank_frame.to_numpy(dtype=float)),
        aspect="auto",
        cmap=score_cmap,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Predictor scores", fontsize=10, fontweight="semibold")
    axes[1].set_xticks(range(len(tool_ids)))
    axes[1].set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")

    labels = work["gene_id"].tolist()
    for axis in axes:
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)

    axes[0].set_ylabel("top positive genes", fontsize=10, color="#3C4858")
    fig.suptitle(
        f"Top {int(positive_fraction * 100)}% of benchmark positives",
        x=0.08,
        y=0.995,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.08,
        0.972,
        (
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} positive genes selected by FDR and effect size"
            "  |  predictor colors show global rank percentile"
        ),
        fontsize=8.6,
        color=NEUTRAL_COLOR,
    )
    fig.colorbar(
        logfc_image,
        ax=axes[0],
        orientation="horizontal",
        fraction=0.08,
        pad=0.08,
        label="logFC",
    )
    fig.colorbar(
        heat,
        ax=axes[1],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        label="global rank percentile",
    )
    _save_figure(fig, out_path)
    return True


def _plot_predictor_correlation_heatmap(
    joined, *, rank_cols, tool_ids, dataset_id, out_path, top_fraction,
):
    ranked = {
        tid: joined[rank_col].astype(float)
        for tid, rank_col in zip(tool_ids, rank_cols)
    }
    top_masks = {tid: _top_fraction_mask(ranked[tid], top_fraction) for tid in tool_ids}

    matrix = pd.DataFrame(index=tool_ids, columns=tool_ids, dtype=float)
    for a in tool_ids:
        for b in tool_ids:
            pair_mask = ranked[a].notna() & ranked[b].notna()
            union_mask = pair_mask & (top_masks[a] | top_masks[b])
            if int(union_mask.sum()) < 2:
                corr = 1.0 if a == b else float("nan")
            else:
                corr = float(ranked[a][union_mask].corr(ranked[b][union_mask], method="spearman"))
            matrix.loc[a, b] = corr

    fig, ax = plt.subplots(
        figsize=(max(5.2, len(tool_ids) * 1.6), max(4.6, len(tool_ids) * 1.25))
    )
    image = ax.imshow(matrix.astype(float).to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tool_ids)))
    ax.set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")
    ax.set_yticks(range(len(tool_ids)))
    ax.set_yticklabels([_tool_label(tool_id) for tool_id in tool_ids])
    ax.set_title(
        f"Predictor agreement (top {int(top_fraction * 100)}%)",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(0.125, 0.955, _dataset_caption(dataset_id), fontsize=9, color=NEUTRAL_COLOR)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0, labelsize=9)
    for i, a in enumerate(tool_ids):
        for j, b in enumerate(tool_ids):
            value = matrix.loc[a, b]
            label = "nan" if pd.isna(value) else f"{value:.2f}"
            color = "white" if not pd.isna(value) and abs(value) >= 0.55 else "#22303C"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman")
    _save_figure(fig, out_path)
    return matrix


def _build_tool_report_markdown(
    *, dataset_id, mirna, cell_line, perturbation, geo_accession,
    de_table_path, joined_tsv,
    tool_id, predictor_output_path, metrics, coverage_info, scatter_png,
):
    lines = [
        f"# Evaluation Report: {dataset_id} | {_tool_label(tool_id)}",
        "",
        "## Dataset",
        f"- dataset_id: `{dataset_id}`",
        f"- mirna: `{mirna or 'NA'}`",
        f"- cell_line: `{cell_line or 'NA'}`",
        f"- perturbation: `{perturbation or 'NA'}`",
        f"- geo_accession: `{geo_accession or 'NA'}`",
        "",
        "## Inputs",
        f"- de_table_path: `{de_table_path or 'NA'}`",
        f"- joined_tsv: `{joined_tsv or 'NA'}`",
        f"- tool_id: `{tool_id}`",
        f"- predictor_output_path: `{predictor_output_path or 'NA'}`",
        "",
        "## Coverage",
        f"- rows_total: `{int(coverage_info['rows_total'])}`",
        f"- rows_scored: `{int(coverage_info['rows_scored'])}`",
        f"- rows_missing_score: `{int(coverage_info['rows_missing_score'])}`",
        f"- coverage: `{coverage_info['coverage']:.6f}`",
        "",
        "## Metrics",
        f"- rows_used: `{int(metrics['rows_used'])}`",
        f"- positives: `{int(metrics['positives'])}`",
        f"- negatives: `{int(metrics['negatives'])}`",
        f"- pearson: `{metrics['pearson']:.6f}`",
        f"- spearman: `{metrics['spearman']:.6f}`",
        f"- aps: `{metrics['aps']:.6f}`",
        f"- pr_auc: `{metrics['pr_auc']:.6f}`",
        f"- auroc: `{metrics['auroc']:.6f}`",
        "",
        "## Plots",
        f"- scatter: `{scatter_png}`",
        "",
    ]
    return "\n".join(lines)


def _render_markdown_pdf(markdown_text, *, pdf_path, title, scatter_png=None):
    def iter_lines():
        for raw_line in markdown_text.splitlines():
            if raw_line.startswith("# "):
                yield {"text": raw_line[2:], "kind": "h1"}
                continue
            if raw_line.startswith("## "):
                yield {"text": raw_line[3:], "kind": "h2"}
                continue
            if raw_line.startswith("- "):
                wrapped = textwrap.wrap(raw_line[2:], width=90) or [""]
                for index, chunk in enumerate(wrapped):
                    prefix = "- " if index == 0 else "  "
                    yield {"text": prefix + chunk, "kind": "body"}
                continue
            if not raw_line.strip():
                yield {"text": "", "kind": "blank"}
                continue
            for chunk in textwrap.wrap(raw_line, width=92) or [""]:
                yield {"text": chunk, "kind": "body"}

    style_map = {
        "h1": {"fontsize": 16, "weight": "bold", "color": "#17324D", "gap": 0.060},
        "h2": {"fontsize": 12, "weight": "bold", "color": "#2F5D8C", "gap": 0.042},
        "body": {"fontsize": 9.5, "weight": "normal", "color": "#22303C", "gap": 0.026},
        "blank": {"fontsize": 9.5, "weight": "normal", "color": "#22303C", "gap": 0.018},
    }

    with PdfPages(pdf_path) as pdf:
        fig = None
        ax = None
        y = 0.0

        def new_page():
            page_fig, page_ax = plt.subplots(figsize=(8.27, 11.69))
            page_ax.axis("off")
            page_fig.patch.set_facecolor("white")
            page_ax.text(
                0.06,
                0.975,
                title,
                fontsize=8.5,
                color=NEUTRAL_COLOR,
                va="top",
                ha="left",
            )
            return page_fig, page_ax, 0.93

        for item in iter_lines():
            style = style_map[item["kind"]]
            if fig is None or y - style["gap"] < 0.06:
                if fig is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                fig, ax, y = new_page()
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
            fig, ax, y = new_page()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if scatter_png is not None and pathlib.Path(scatter_png).is_file():
            image = plt.imread(scatter_png)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            fig.patch.set_facecolor("white")
            ax.text(
                0.06,
                0.975,
                f"{title} | Scatter plot",
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


def _write_tool_report(
    *, dataset_id, mirna, cell_line, perturbation, geo_accession,
    de_table_path, joined_tsv,
    tool_id, predictor_output_path, metrics, markdown_path, pdf_path, coverage_info, scatter_png,
):
    markdown_text = _build_tool_report_markdown(
        dataset_id=dataset_id,
        mirna=mirna,
        cell_line=cell_line,
        perturbation=perturbation,
        geo_accession=geo_accession,
        de_table_path=de_table_path,
        joined_tsv=joined_tsv,
        tool_id=tool_id,
        predictor_output_path=predictor_output_path,
        metrics=metrics,
        coverage_info=coverage_info,
        scatter_png=scatter_png,
    )
    markdown_path.write_text(markdown_text + "\n", encoding="utf-8")
    _render_markdown_pdf(
        markdown_text,
        pdf_path=pdf_path,
        title=f"{dataset_id} | {_tool_label(tool_id)}",
        scatter_png=scatter_png,
    )


def evaluate_joined_dataframe(
    joined, *, plots_dir, reports_dir,
    fdr_threshold, abs_logfc_threshold, predictor_top_fraction,
    dataset_id=None, mirna=None, cell_line=None,
    perturbation=None, geo_accession=None,
    de_table_path=None, joined_tsv=None,
    predictor_output_paths=None,
    logger=None,
):
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    score_cols = sorted(c for c in joined.columns if c.startswith(SCORE_PREFIX))
    if not score_cols:
        raise ValueError("No score_<tool_id> columns found in joined dataframe.")

    dataset_id = dataset_id or (
        str(joined["dataset_id"].iloc[0]) if "dataset_id" in joined.columns else "NA"
    )
    mirna = mirna or (
        str(joined["mirna"].iloc[0]) if "mirna" in joined.columns else None
    )
    dataset_plots_dir = plots_dir / dataset_id
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)
    tool_ids = [_tool_id_from_score_col(sc) for sc in score_cols]
    rank_cols = []
    for score_col, tool_id in zip(score_cols, tool_ids):
        rank_col = _rank_col_for_tool(tool_id)
        if rank_col not in joined.columns:
            joined = joined.copy()
            joined[rank_col] = _rank_scale_scores(joined[score_col])
        rank_cols.append(rank_col)

    metric_rows = []
    dataset_plots = {}
    predictor_correlation_tsv = None
    comparisons = []

    _emit_log(logger, f"    Evaluation start: {dataset_id} | tools={tool_ids}")

    for score_col, tool_id in zip(score_cols, tool_ids):
        _emit_log(logger, f"    Tool: {tool_id} | preparing scored pairs")
        scored, coverage_info = _prepare_scored_frame(
            joined, score_col=score_col,
            fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
        )
        scatter_png = dataset_plots_dir / f"{tool_id}_score_vs_logFC.png"
        gsea_png = dataset_plots_dir / f"{tool_id}_gsea_enrichment.png"
        pearson, spearman = _plot_scatter_with_correlation(
            scored,
            score_col=score_col,
            dataset_id=dataset_id,
            tool_id=tool_id,
            out_path=scatter_png,
        )
        enrichment_score = _plot_gsea_enrichment(
            scored,
            score_col=score_col,
            dataset_id=dataset_id,
            tool_id=tool_id,
            out_path=gsea_png,
        )
        pr_auc, aps = _compute_pr_metrics(scored["is_positive"], scored[score_col])
        auroc = _compute_auroc(scored["is_positive"], scored[score_col])

        report_md = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.md"
        report_pdf = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.pdf"
        metrics = {
            "rows_total": float(coverage_info["rows_total"]),
            "rows_used": float(len(scored)),
            "positives": float(scored["is_positive"].sum()),
            "negatives": float(len(scored) - int(scored["is_positive"].sum())),
            "rows_missing_score": float(coverage_info["rows_missing_score"]),
            "coverage": float(coverage_info["coverage"]),
            "pearson": pearson, "spearman": spearman,
            "aps": aps, "pr_auc": pr_auc, "auroc": auroc,
        }
        _write_tool_report(
            dataset_id=dataset_id, mirna=mirna, cell_line=cell_line,
            perturbation=perturbation, geo_accession=geo_accession,
            de_table_path=de_table_path, joined_tsv=joined_tsv,
            tool_id=tool_id,
            predictor_output_path=(predictor_output_paths or {}).get(tool_id),
            metrics=metrics, markdown_path=report_md, pdf_path=report_pdf,
            coverage_info=coverage_info, scatter_png=scatter_png,
        )
        _emit_log(
            logger,
            (
                f"    Tool: {tool_id} | coverage={coverage_info['coverage']:.1%} "
                f"| rows={coverage_info['rows_scored']}/{coverage_info['rows_total']} "
                f"| APS={aps:.3f} | AUROC={auroc:.3f} | ES={enrichment_score:.3f}"
            ),
        )
        _emit_log(logger, f"    Tool: {tool_id} | wrote scatter/report")

        metric_rows.append({
            "dataset_id": dataset_id, "mirna": mirna, "cell_line": cell_line,
            "perturbation": perturbation, "geo_accession": geo_accession,
            "tool_id": tool_id,
            "rows_total": coverage_info["rows_total"],
            "rows_scored": coverage_info["rows_scored"],
            "rows_missing_score": coverage_info["rows_missing_score"],
            "coverage": coverage_info["coverage"],
            "aps": aps, "spearman": spearman, "auroc": auroc, "pr_auc": pr_auc,
        })
        comparisons.append({
            "tool_id": tool_id,
            "y_true": scored["is_positive"],
            "y_score": scored[score_col],
            "coverage": coverage_info["coverage"],
        })
        dataset_plots[f"{tool_id}_scatter"] = str(scatter_png)
        dataset_plots[f"{tool_id}_gsea_enrichment"] = str(gsea_png)

    heatmap_png = dataset_plots_dir / "algorithms_vs_genes_heatmap.png"
    _plot_algorithms_vs_genes_heatmap(
        joined, score_cols=score_cols, rank_cols=rank_cols, tool_ids=tool_ids,
        dataset_id=dataset_id, out_path=heatmap_png,
        fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
    )
    dataset_plots["algorithms_vs_genes_heatmap"] = str(heatmap_png)
    _emit_log(logger, f"    Dataset: {dataset_id} | wrote gene-level heatmap")

    top_positive_heatmap_png = dataset_plots_dir / "top_10pct_positive_heatmap.png"
    wrote_top_positive_heatmap = _plot_top_positive_heatmap(
        joined,
        rank_cols=rank_cols,
        tool_ids=tool_ids,
        dataset_id=dataset_id,
        out_path=top_positive_heatmap_png,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        positive_fraction=0.10,
    )
    if wrote_top_positive_heatmap:
        dataset_plots["top_10pct_positive_heatmap"] = str(top_positive_heatmap_png)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote top-positive heatmap")

    if len(score_cols) >= 2:
        comparison_pr_png = dataset_plots_dir / "predictor_pr_curves.png"
        comparison_roc_png = dataset_plots_dir / "predictor_roc_curves.png"
        _plot_predictor_pr_curves(
            comparisons,
            dataset_id=dataset_id,
            out_path=comparison_pr_png,
        )
        _plot_predictor_roc_curves(
            comparisons,
            dataset_id=dataset_id,
            out_path=comparison_roc_png,
        )
        dataset_plots["predictor_pr_curves"] = str(comparison_pr_png)
        dataset_plots["predictor_roc_curves"] = str(comparison_roc_png)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote PR/ROC comparison plots")

        corr_png = dataset_plots_dir / "predictor_correlation_heatmap.png"
        corr_tsv = reports_dir / f"{dataset_id}__predictor_correlation.tsv"
        corr_matrix = _plot_predictor_correlation_heatmap(
            joined, rank_cols=rank_cols, tool_ids=tool_ids,
            dataset_id=dataset_id, out_path=corr_png,
            top_fraction=predictor_top_fraction,
        )
        corr_matrix.to_csv(corr_tsv, sep="\t")
        dataset_plots["predictor_correlation_heatmap"] = str(corr_png)
        predictor_correlation_tsv = str(corr_tsv)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote predictor correlation outputs")

    _emit_log(logger, f"    Evaluation complete: {dataset_id}")

    return {
        "metric_rows": metric_rows,
        "plots": dataset_plots,
        "predictor_correlation_tsv": predictor_correlation_tsv,
        "tool_ids": tool_ids,
        "score_cols": score_cols,
    }


def write_metric_tables(metric_rows, tables_dir, *, logger=None):
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise ValueError("No metric rows were produced.")

    id_cols = ["dataset_id", "mirna", "cell_line", "perturbation", "geo_accession"]
    metrics_df[id_cols] = metrics_df[id_cols].fillna("NA")
    out_paths = {}
    for metric_name, filename in [
        ("coverage", "coverage_per_experiment.tsv"),
        ("aps", "aps_per_experiment.tsv"),
        ("pr_auc", "pr_auc_per_experiment.tsv"),
        ("spearman", "spearman_per_experiment.tsv"),
        ("auroc", "auroc_per_experiment.tsv"),
    ]:
        wide = metrics_df.pivot_table(
            index=id_cols, columns="tool_id", values=metric_name, aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        out_path = tables_dir / filename
        wide.to_csv(out_path, sep="\t", index=False)
        out_paths[metric_name] = str(out_path)
        _emit_log(logger, f"  Wrote {metric_name} table: {out_path}")
    return out_paths


def _plot_cross_dataset_metric_heatmap(summary_df, *, metric_names, out_path):
    tool_ids = summary_df["tool_id"].tolist()
    matrix = summary_df[[f"{metric_name}_mean" for metric_name in metric_names]].to_numpy(dtype=float).T

    fig, ax = plt.subplots(
        figsize=(max(5.6, len(tool_ids) * 1.25), max(4.8, len(metric_names) * 1.0))
    )
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tool_ids)))
    ax.set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels([metric_name.upper() for metric_name in metric_names])
    ax.set_title(
        "Cross-dataset mean metric summary",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0, labelsize=9)
    for row_index, metric_name in enumerate(metric_names):
        for col_index, _tool_id in enumerate(tool_ids):
            value = summary_df.iloc[col_index][f"{metric_name}_mean"]
            color = "white" if value >= 0.55 else "#22303C"
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mean value")
    _save_figure(fig, out_path)


def _plot_cross_dataset_metric_distributions(metrics_df, *, metric_names, out_path):
    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(7.2, max(7.2, len(metric_names) * 2.1)),
        sharex=False,
        gridspec_kw={"hspace": 0.4},
    )
    if len(metric_names) == 1:
        axes = [axes]

    tool_ids = list(metrics_df["tool_id"].drop_duplicates())
    positions = np.arange(len(tool_ids), dtype=float)

    for metric_index, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        _style_axes(ax, grid_axis="y")
        data = []
        for tool_id in tool_ids:
            values = metrics_df.loc[metrics_df["tool_id"] == tool_id, metric_name].dropna().astype(float).tolist()
            data.append(values)
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], CURVE_COLORS * ((len(tool_ids) // len(CURVE_COLORS)) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for median in box["medians"]:
            median.set_color("#22303C")
            median.set_linewidth(1.4)
        for whisker in box["whiskers"]:
            whisker.set_color("#7A8798")
        for cap in box["caps"]:
            cap.set_color("#7A8798")

        for tool_index, values in enumerate(data):
            if not values:
                continue
            jitter = np.linspace(-0.09, 0.09, num=len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), positions[tool_index]) + jitter,
                values,
                s=18,
                alpha=0.75,
                color=CURVE_COLORS[tool_index % len(CURVE_COLORS)],
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

        ax.set_ylim(0, 1.02)
        ax.set_ylabel(metric_name.upper(), fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")

    axes[0].set_title(
        "Cross-dataset metric distributions",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    _save_figure(fig, out_path)


def _plot_coverage_vs_performance(summary_df, *, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=False)
    metric_specs = [("aps_mean", "Mean APS"), ("auroc_mean", "Mean AUROC")]
    coverage = summary_df["coverage_mean"].astype(float).to_numpy()

    for ax, (metric_col, metric_label) in zip(axes, metric_specs):
        _style_axes(ax, grid_axis="both")
        values = summary_df[metric_col].astype(float).to_numpy()
        for index, tool_id in enumerate(summary_df["tool_id"].tolist()):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            ax.scatter(
                coverage[index],
                values[index],
                s=80,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.text(
                coverage[index] + 0.01,
                values[index],
                _tool_label(tool_id),
                fontsize=8.5,
                color="#22303C",
                va="center",
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean coverage", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
    axes[0].set_title(
        "Coverage vs performance",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    _save_figure(fig, out_path)


def _plot_rank_class_distributions(joined_frames, *, out_path, fdr_threshold, abs_logfc_threshold):
    combined = pd.concat(joined_frames, ignore_index=True)
    rank_cols = sorted(col for col in combined.columns if col.startswith(GLOBAL_RANK_PREFIX))
    if not rank_cols:
        return False

    work = combined[["logFC", "FDR", *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    if work.empty:
        return False
    work["logFC"] = work["logFC"].astype(float)
    work["FDR"] = work["FDR"].astype(float)
    work["abs_logFC"] = work["logFC"].abs()
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold) & (work["abs_logFC"] > abs_logfc_threshold)
    ).astype(int)

    tool_ids = [_tool_id_from_score_col(rank_col.replace(GLOBAL_RANK_PREFIX, SCORE_PREFIX, 1)) for rank_col in rank_cols]
    positive_data = [work.loc[work["is_positive"] == 1, rank_col].dropna().astype(float).tolist() for rank_col in rank_cols]
    background_data = [work.loc[work["is_positive"] == 0, rank_col].dropna().astype(float).tolist() for rank_col in rank_cols]
    if not any(positive_data) or not any(background_data):
        return False

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True, gridspec_kw={"hspace": 0.25})
    panel_specs = [
        (axes[0], positive_data, "GT positives"),
        (axes[1], background_data, "Background genes"),
    ]
    positions = np.arange(len(tool_ids), dtype=float)
    colors = [CURVE_COLORS[index % len(CURVE_COLORS)] for index in range(len(tool_ids))]

    for ax, panel_data, panel_title in panel_specs:
        _style_axes(ax, grid_axis="y")
        box = ax.boxplot(
            panel_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for median in box["medians"]:
            median.set_color("#22303C")
            median.set_linewidth(1.4)
        for whisker in box["whiskers"]:
            whisker.set_color("#7A8798")
        for cap in box["caps"]:
            cap.set_color("#7A8798")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Global rank", fontsize=10)
        ax.set_title(panel_title, fontsize=10.5, fontweight="semibold", loc="left", pad=10)

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")
    axes[0].set_title(
        "Positive vs background rank distributions",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    _save_figure(fig, out_path)
    return True


def write_cross_dataset_summaries(
    metric_rows,
    tables_dir,
    plots_dir,
    *,
    joined_frames=None,
    fdr_threshold=0.05,
    abs_logfc_threshold=1.0,
    logger=None,
):
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        return {
            "tables": {},
            "plots": {},
        }

    metric_names = ["coverage", "aps", "pr_auc", "spearman", "auroc"]
    summary = metrics_df.groupby("tool_id")[metric_names].agg(["count", "mean", "median", "std", "min", "max"])
    summary.columns = [f"{metric_name}_{stat_name}" for metric_name, stat_name in summary.columns]
    summary = summary.reset_index()
    summary_path = tables_dir / "cross_dataset_predictor_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    _emit_log(logger, f"  Wrote cross-dataset summary table: {summary_path}")

    heatmap_path = plots_dir / "cross_dataset_metric_heatmap.png"
    _plot_cross_dataset_metric_heatmap(summary, metric_names=metric_names, out_path=heatmap_path)
    _emit_log(logger, f"  Wrote cross-dataset metric heatmap: {heatmap_path}")

    distributions_path = plots_dir / "cross_dataset_metric_distributions.png"
    _plot_cross_dataset_metric_distributions(metrics_df, metric_names=metric_names, out_path=distributions_path)
    _emit_log(logger, f"  Wrote cross-dataset metric distributions: {distributions_path}")

    coverage_scatter_path = plots_dir / "coverage_vs_performance.png"
    _plot_coverage_vs_performance(summary, out_path=coverage_scatter_path)
    _emit_log(logger, f"  Wrote coverage vs performance plot: {coverage_scatter_path}")

    rank_distribution_path = plots_dir / "positive_background_rank_distributions.png"
    wrote_rank_distributions = False
    if joined_frames:
        wrote_rank_distributions = _plot_rank_class_distributions(
            joined_frames,
            out_path=rank_distribution_path,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
        )
        if wrote_rank_distributions:
            _emit_log(logger, f"  Wrote rank distribution plot: {rank_distribution_path}")

    return {
        "tables": {
            "cross_dataset_predictor_summary": str(summary_path),
        },
        "plots": {
            "cross_dataset_metric_heatmap": str(heatmap_path),
            "cross_dataset_metric_distributions": str(distributions_path),
            "coverage_vs_performance": str(coverage_scatter_path),
            **(
                {"positive_background_rank_distributions": str(rank_distribution_path)}
                if wrote_rank_distributions
                else {}
            ),
        },
    }
