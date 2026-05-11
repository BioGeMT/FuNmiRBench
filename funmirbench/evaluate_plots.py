"""Plot helpers for per-dataset predictor evaluation."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from funmirbench.evaluate_common import *


def _plot_scatter_with_correlation(df, *, score_col, dataset_id, tool_id, positives_total, out_path):
    pearson = float(df[score_col].corr(df["expected_effect"], method="pearson"))
    spearman = float(df[score_col].corr(df["expected_effect"], method="spearman"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    positive_mask = df["is_positive"].astype(bool)
    negatives = df.loc[~positive_mask]
    positives = df.loc[positive_mask]
    predictor_color = _tool_color(tool_id)

    _style_axes(ax, grid_axis="both")
    ax.scatter(
        negatives[score_col],
        negatives["expected_effect"],
        s=18,
        alpha=0.55,
        color=NEGATIVE_COLOR,
        edgecolors="none",
        label="background genes",
        rasterized=True,
    )
    ax.scatter(
        positives[score_col],
        positives["expected_effect"],
        s=28,
        alpha=0.9,
        color=predictor_color,
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
    ax.set_ylabel("Expected target effect", fontsize=10)
    ax.set_title(
        f"{_tool_label(tool_id)} score vs expected effect",
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
            f"  |  positives={_positive_count_caption(int(positive_mask.sum()), positives_total)}"
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


def _plot_gsea_enrichment(df, *, score_col, dataset_id, tool_id, positives_total, out_path):
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
        2,
        1,
        figsize=(7.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.6, 0.9], "hspace": 0.08},
    )
    curve_ax, hit_ax = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#3C4858", labelsize=9)

    curve_ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    curve_ax.set_axisbelow(True)
    predictor_color = _tool_color(tool_id)
    curve_ax.plot(positions, running_es, color=predictor_color, linewidth=2.2)
    curve_ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle="--", alpha=0.8)
    curve_ax.scatter(
        [positions[peak_index]],
        [running_es[peak_index]],
        color="black",
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
            f"  |  positives={_positive_count_caption(total_hits, positives_total)}"
            f"  |  ES={es:.3f}"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )

    hit_ax.axhspan(0.0, 1.0, color="#E5E7EB", alpha=1.0, zorder=0)
    if len(hit_positions):
        hit_start = float(hit_positions.min()) - 0.5
        hit_end = float(hit_positions.max()) + 0.5
        hit_ax.add_patch(
            Rectangle(
                (hit_start, 0.0),
                hit_end - hit_start,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.1,
                zorder=2,
            )
        )
    hit_ax.vlines(hit_positions, 0.0, 1.0, color=POSITIVE_COLOR, linewidth=0.9, zorder=3)
    hit_ax.set_ylim(0.0, 1.0)
    hit_ax.set_yticks([])
    hit_ax.set_ylabel("Hits", fontsize=9)
    hit_ax.spines["left"].set_visible(False)
    hit_ax.set_xlabel("Ranked genes", fontsize=10)

    _save_figure(fig, out_path)
    return es


def _compute_pr_metrics(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    aps = average_precision_score(y_true, y_score)
    return float(pr_auc), float(aps)


def _compute_auroc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))


def _plot_single_predictor_pr_curve(item, *, dataset_id, out_path):
    precision, recall, _ = precision_recall_curve(item["y_true"], item["y_score"])
    pr_auc = auc(recall, precision)
    baseline = float(item["y_true"].mean())
    predictor_color = _tool_color(item["tool_id"])

    fig, ax = plt.subplots(figsize=(6.1, 4.9))
    _style_axes(ax, grid_axis="both")
    ax.plot(
        recall,
        precision,
        linewidth=2.2,
        color=predictor_color,
        label=f"PR-AUC {pr_auc:.3f}",
    )
    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.4,
        color=NEUTRAL_COLOR,
        label=f"baseline {baseline:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(
        f"{_tool_label(item['tool_id'])} precision-recall",
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
            f"  |  n={len(item['y_true']):,}"
            f"  |  positives={_positive_count_caption(int(item['y_true'].sum()), item['positives_total'])}"
            f"  |  cov={item['coverage']:.0%}"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    _save_figure(fig, out_path)


def _plot_single_predictor_roc_curve(item, *, dataset_id, out_path):
    fpr, tpr, _ = roc_curve(item["y_true"], item["y_score"])
    auroc = roc_auc_score(item["y_true"], item["y_score"])
    predictor_color = _tool_color(item["tool_id"])

    fig, ax = plt.subplots(figsize=(6.1, 4.9))
    _style_axes(ax, grid_axis="both")
    ax.plot(
        fpr,
        tpr,
        linewidth=2.2,
        color=predictor_color,
        label=f"AUROC {auroc:.3f}",
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
        f"{_tool_label(item['tool_id'])} ROC",
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
            f"  |  n={len(item['y_true']):,}"
            f"  |  positives={_positive_count_caption(int(item['y_true'].sum()), item['positives_total'])}"
            f"  |  cov={item['coverage']:.0%}"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")
    _save_figure(fig, out_path)


def _plot_predictor_pr_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    _style_axes(ax, grid_axis="both")
    baseline = None
    for item in comparisons:
        precision, recall, _ = precision_recall_curve(item["y_true"], item["y_score"])
        pr_auc = auc(recall, precision)
        if baseline is None:
            baseline = float(item["y_true"].mean())
        ax.plot(
            recall,
            precision,
            label=(
                f"{_tool_label(item['tool_id'])} "
                f"({pr_auc:.3f})"
            ),
            linewidth=2.2,
            color=_tool_color(item["tool_id"]),
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1.4,
            color=NEUTRAL_COLOR,
            label=f"baseline ({baseline:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(
        "Precision-recall comparison (common scored pairs)",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    comparison_size = len(comparisons[0]["y_true"]) if comparisons else 0
    fig.text(
        0.125,
        0.955,
        f"{_dataset_caption(dataset_id)}  |  common n={comparison_size:,}",
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_predictor_pr_curves_own_scored(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    _style_axes(ax, grid_axis="both")
    for item in comparisons:
        precision, recall, _ = precision_recall_curve(item["y_true"], item["y_score"])
        pr_auc = auc(recall, precision)
        baseline = float(item["y_true"].mean())
        ax.plot(
            recall,
            precision,
            label=(
                f"{_tool_label(item['tool_id'])} "
                f"({pr_auc:.3f}, base {baseline:.3f})"
            ),
            linewidth=2.2,
            color=_tool_color(item["tool_id"]),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(
        "Precision-recall comparison (each predictor's scored pairs)",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(
        0.125,
        0.955,
        f"{_dataset_caption(dataset_id)}  |  own scored sets",
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(
        frameon=False,
        fontsize=8.4,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _prepare_common_scored_frame(
    joined, *, score_cols, fdr_threshold, abs_logfc_threshold, perturbation=None,
):
    required_cols = {"gene_id", "logFC", "FDR", *score_cols}
    missing = [col for col in required_cols if col not in joined.columns]
    if missing:
        raise ValueError(f"Joined table missing required columns: {missing}")

    keep_cols = ["gene_id", "logFC", "FDR", *score_cols]
    for optional in ("dataset_id", "mirna", "perturbation", "PValue"):
        if optional in joined.columns:
            keep_cols.append(optional)

    keep = joined[keep_cols].copy()
    keep = keep[keep["logFC"].notna() & keep["FDR"].notna()].copy()
    keep = keep[keep["FDR"].astype(float) > 0].copy()
    if keep.empty:
        raise ValueError("No usable rows remain for common PR comparison.")

    keep = _annotate_ground_truth(keep, perturbation=perturbation)
    keep["is_positive"] = _positive_mask(
        keep,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
    keep = keep.dropna(subset=score_cols).copy()
    if keep.empty:
        raise ValueError("No common scored rows remain for PR comparison.")

    positives = int(keep["is_positive"].sum())
    negatives = int(len(keep) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Common PR comparison needs both positives and negatives.")
    for score_col in score_cols:
        keep[score_col] = keep[score_col].astype(float)
    return keep


def _plot_predictor_roc_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    _style_axes(ax, grid_axis="both")
    for item in comparisons:
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_score"])
        auroc = roc_auc_score(item["y_true"], item["y_score"])
        ax.plot(
            fpr,
            tpr,
            label=f"{_tool_label(item['tool_id'])} ({auroc:.3f})",
            linewidth=2.2,
            color=_tool_color(item["tool_id"]),
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
        "ROC comparison (common scored pairs)",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    comparison_size = len(comparisons[0]["y_true"]) if comparisons else 0
    fig.text(
        0.125,
        0.955,
        f"{_dataset_caption(dataset_id)}  |  common n={comparison_size:,}",
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_predictor_roc_curves_own_scored(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    _style_axes(ax, grid_axis="both")
    for item in comparisons:
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_score"])
        auroc = roc_auc_score(item["y_true"], item["y_score"])
        baseline = float(item["y_true"].mean())
        ax.plot(
            fpr,
            tpr,
            label=(
                f"{_tool_label(item['tool_id'])} "
                f"({auroc:.3f}, base {baseline:.3f})"
            ),
            linewidth=2.2,
            color=_tool_color(item["tool_id"]),
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
        "ROC comparison (each predictor's scored pairs)",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(
        0.125,
        0.955,
        f"{_dataset_caption(dataset_id)}  |  own scored sets",
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(
        frameon=False,
        fontsize=8.4,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_predictor_gsea_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    _style_axes(ax, grid_axis="both")
    for item in comparisons:
        order_frame = {"y_true": item["y_true"], "y_score": item["y_score"]}
        sort_cols = ["y_score"]
        ascending = [False]
        if "gene_id" in item:
            order_frame["gene_id"] = item["gene_id"]
            sort_cols.append("gene_id")
            ascending.append(True)
        ordered = pd.DataFrame(order_frame).sort_values(
            sort_cols, ascending=ascending, kind="mergesort"
        ).reset_index(drop=True)
        hits = ordered["y_true"].astype(int).to_numpy(dtype=int)
        total_hits = int(hits.sum())
        total_misses = int(len(hits) - total_hits)
        if total_hits == 0 or total_misses == 0:
            continue
        hit_step = 1.0 / total_hits
        miss_step = 1.0 / total_misses
        running_es = np.cumsum(np.where(hits == 1, hit_step, -miss_step))
        es = float(running_es.max())
        min_es = float(running_es.min())
        if abs(min_es) > abs(es):
            es = min_es
        ax.plot(
            np.arange(1, len(ordered) + 1),
            running_es,
            label=f"{_tool_label(item['tool_id'])} (ES {es:.3f})",
            linewidth=2.1,
            color=_tool_color(item["tool_id"]),
        )
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Ranked genes", fontsize=10)
    ax.set_ylabel("Running ES", fontsize=10)
    ax.set_title(
        "GSEA comparison (common scored pairs)",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    comparison_size = len(comparisons[0]["y_true"]) if comparisons else 0
    fig.text(
        0.125,
        0.955,
        f"{_dataset_caption(dataset_id)}  |  common n={comparison_size:,}",
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_top_prediction_effect_cdfs(
    joined, *, score_cols, tool_ids, dataset_id, out_path, top_n=TOP_PREDICTION_CDF_N,
    perturbation=None,
):
    required_cols = {"logFC", *score_cols}
    missing = [col for col in required_cols if col not in joined.columns]
    if missing:
        raise ValueError(f"Joined table missing required columns: {missing}")

    keep_cols = ["logFC", "FDR", *score_cols]
    if "gene_id" in joined.columns:
        keep_cols.append("gene_id")
    work = joined[keep_cols].copy()
    work = work[work["logFC"].notna()].copy()
    work = _annotate_ground_truth(work, perturbation=perturbation)
    if work.empty:
        raise ValueError("No usable rows remain for top-prediction effect CDF plot.")

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    _style_axes(ax, grid_axis="both")

    background_x, background_y = _ecdf(work["expected_effect"])
    ax.step(
        background_x,
        background_y,
        where="post",
        color=NEUTRAL_COLOR,
        linewidth=1.7,
        linestyle="--",
        alpha=0.9,
        label=f"All genes (n={len(work):,})",
    )

    medians = []
    for score_col, tool_id in zip(score_cols, tool_ids):
        scored = work[work[score_col].notna()].copy()
        if scored.empty:
            continue
        sort_cols = [score_col]
        ascending = [False]
        if "gene_id" in scored.columns:
            sort_cols.append("gene_id")
            ascending.append(True)
        scored = scored.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        top_count = min(int(top_n), len(scored))
        top_values = scored["expected_effect"].head(top_count).to_numpy(dtype=float)
        median_effect = float(np.nanmedian(top_values))
        medians.append(median_effect)
        x_values, y_values = _ecdf(top_values)
        ax.step(
            x_values,
            y_values,
            where="post",
            color=_tool_color(tool_id),
            linewidth=2.2,
            label=f"{_tool_label(tool_id)} top {top_count} | med {median_effect:.2f}",
        )

    ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle=":", alpha=0.8)
    if medians:
        ax.axvline(
            float(np.nanmedian(work["expected_effect"])),
            color="#7A8798",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            label="all-gene median",
        )
        limit = _nice_symmetric_limit(
            np.concatenate([work["expected_effect"].to_numpy(dtype=float), np.asarray(medians)]),
            floor=1.0,
        )
        ax.set_xlim(-limit, limit)
    ax.set_xlabel("Perturbation-aware effect", fontsize=10)
    ax.set_ylabel("Cumulative fraction", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Top prediction effect CDFs",
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
            f"  |  top {int(top_n)} per predictor"
            "  |  higher values indicate stronger perturbation-consistent effect"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(
        frameon=False,
        fontsize=8.6,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    _save_figure(fig, out_path)


def _plot_algorithms_vs_genes_heatmap(
    joined, *, score_cols, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, perturbation=None,
):
    work = joined[["gene_id", "logFC", "FDR", *score_cols, *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work = _annotate_ground_truth(work, perturbation=perturbation)
    work["is_positive"] = _positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
    work = _sort_heatmap_rows_by_logfc(work)

    rank_frame = pd.DataFrame(
        {
            tool_id: work[rank_col].astype(float)
            for tool_id, rank_col in zip(tool_ids, rank_cols)
        }
    )

    max_abs_logfc = _nice_symmetric_limit(work["logFC"].to_numpy(dtype=float), floor=1.0)
    figure_height = max(5.6, min(12, 0.025 * len(work)))
    figure_width = max(10.5, 5.2 + len(tool_ids) * 1.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.5, 0.6, max(2.8, len(tool_ids) * 1.25)]},
    )
    fig.subplots_adjust(top=0.86, bottom=0.2, wspace=0.22)
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
    axes[0].set_title("GT status", fontsize=10, fontweight="semibold")
    axes[0].set_xticks([])

    logfc_image = axes[1].imshow(
        work["logFC"].to_numpy().reshape(-1, 1), aspect="auto", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs_logfc, vcenter=0.0, vmax=max_abs_logfc),
        interpolation="nearest",
    )
    axes[1].set_title("logFC", fontsize=10, fontweight="semibold")
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
    axes[2].set_title("Predictor scores", fontsize=10, fontweight="semibold")
    axes[2].set_xticks(range(len(tool_ids)))
    axes[2].set_xticklabels(
        [_wrap_axis_label(_tool_label(tool_id)) for tool_id in tool_ids],
        rotation=30,
        ha="right",
    )

    if len(work) <= 40:
        labels = work["gene_id"].tolist()
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[1].set_yticks([])
        axes[2].set_yticks([])
    else:
        for axis in axes:
            axis.set_yticks([])

    axes[0].set_ylabel("genes ranked by logFC", fontsize=10, color="#3C4858")
    _add_figure_heading(
        fig,
        title="Gene-level benchmarking overview",
        subtitle=(
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} genes ordered by perturbation-aware logFC"
            "  |  dark GT = benchmark positive  |  gray cells indicate missing predictor pairs"
        ),
        title_y=0.975,
        subtitle_y=0.935,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=logfc_image,
        anchor_ax=axes[1],
        label="observed logFC",
        ticks=[-max_abs_logfc, 0.0, max_abs_logfc],
        height=0.014,
        pad=0.055,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=heat,
        anchor_ax=axes[2],
        label="dataset-local rank percentile",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        height=0.014,
        pad=0.055,
    )
    _save_figure(fig, out_path)


def _plot_top_positive_heatmap(
    joined, *, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, positive_fraction, perturbation=None,
):
    work = joined[["gene_id", "logFC", "FDR", *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work = _annotate_ground_truth(work, perturbation=perturbation)
    work["is_positive"] = _positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
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
    figure_height = max(4.2, min(9.0, 0.22 * len(work)))
    figure_width = max(10.5, 5.2 + len(tool_ids) * 1.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.7, max(2.8, len(tool_ids) * 1.25)]},
    )
    fig.subplots_adjust(top=0.84, bottom=0.28, wspace=0.2)
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
    axes[0].set_title("logFC", fontsize=10, fontweight="semibold")
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
    axes[1].set_title("Predictor scores", fontsize=10, fontweight="semibold")
    axes[1].set_xticks(range(len(tool_ids)))
    axes[1].set_xticklabels(
        [_wrap_axis_label(_tool_label(tool_id)) for tool_id in tool_ids],
        rotation=30,
        ha="right",
    )

    labels = work["gene_id"].tolist()
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[1].set_yticks([])

    axes[0].set_ylabel("top positives ranked by logFC", fontsize=10, color="#3C4858")
    _add_figure_heading(
        fig,
        title=f"Top {int(positive_fraction * 100)}% of benchmark positives",
        subtitle=(
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} positive genes {_selection_caption(fdr_threshold)}"
            "  |  rows ordered by perturbation-aware logFC"
        ),
        title_y=0.975,
        subtitle_y=0.935,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=logfc_image,
        anchor_ax=axes[0],
        label="observed logFC",
        ticks=[-max_abs_logfc, 0.0, max_abs_logfc],
        height=0.016,
        pad=0.06,
    )
    _add_horizontal_colorbar(
        fig,
        mappable=heat,
        anchor_ax=axes[1],
        label="dataset-local rank percentile",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        height=0.016,
        pad=0.06,
    )
    _save_figure(fig, out_path)
    return True


def _build_predictor_correlation_matrix(
    joined, *, rank_cols, tool_ids, top_fraction,
):
    ranked = {
        tid: joined[rank_col].astype(float)
        for tid, rank_col in zip(tool_ids, rank_cols)
    }
    tie_breaker = joined["gene_id"] if "gene_id" in joined.columns else None
    top_masks = {
        tid: _top_fraction_mask(ranked[tid], top_fraction, tie_breaker=tie_breaker)
        for tid in tool_ids
    }

    matrix = pd.DataFrame(index=tool_ids, columns=tool_ids, dtype=float)
    for a in tool_ids:
        for b in tool_ids:
            pair_mask = ranked[a].notna() & ranked[b].notna()
            shared_top_mask = pair_mask & top_masks[a] & top_masks[b]
            if int(shared_top_mask.sum()) < 2:
                corr = 1.0 if a == b else float("nan")
            else:
                corr = float(
                    ranked[a][shared_top_mask].corr(
                        ranked[b][shared_top_mask], method="spearman"
                    )
                )
            matrix.loc[a, b] = corr

    return matrix


__all__ = [name for name in globals() if not name.startswith("__")]
