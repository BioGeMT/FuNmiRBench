"""Evaluate joined GT/prediction tables: metrics, plots, reports."""

import math
import os
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
LOCAL_RANK_PREFIX = "local_rank_"
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


def _metric_plot_limits(metric_name):
    if metric_name == "spearman":
        return -1.02, 1.02
    return 0.0, 1.02


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


def _normalize_perturbation(value):
    text = str(value or "").strip().upper()
    if text in {"", "NAN", "NONE", "<NA>"}:
        return ""
    return text


def _infer_perturbation_from_dataset_id(value):
    dataset_id = str(value or "").upper()
    for perturbation in ("OE", "KO", "KD"):
        if f"_{perturbation}_" in dataset_id:
            return perturbation
    return ""


def _resolve_perturbation_series(df, *, perturbation=None):
    fallback = _normalize_perturbation(perturbation)
    if "perturbation" in df.columns:
        resolved = df["perturbation"].astype(str).map(_normalize_perturbation)
    elif "dataset_id" in df.columns:
        resolved = df["dataset_id"].astype(str).map(_infer_perturbation_from_dataset_id)
    else:
        resolved = pd.Series("", index=df.index, dtype="object")
    if fallback:
        resolved = resolved.where(resolved != "", fallback)
    return resolved.fillna("")


def _expected_effect_from_logfc(logfc, perturbations):
    expected_effect = logfc.abs().astype(float)
    oe_mask = perturbations == "OE"
    ko_mask = perturbations.isin(["KO", "KD"])
    expected_effect.loc[oe_mask] = -logfc.loc[oe_mask]
    expected_effect.loc[ko_mask] = logfc.loc[ko_mask]
    return expected_effect


def _annotate_ground_truth(df, *, perturbation=None):
    out = df.copy()
    out["logFC"] = out["logFC"].astype(float)
    out["FDR"] = out["FDR"].astype(float)
    out["abs_logFC"] = out["logFC"].abs()
    out["neglog10_FDR"] = _safe_neglog10(out["FDR"])
    out["resolved_perturbation"] = _resolve_perturbation_series(out, perturbation=perturbation)
    out["expected_effect"] = _expected_effect_from_logfc(
        out["logFC"],
        out["resolved_perturbation"],
    )
    return out


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


def _rank_col_for_tool(tool_id, *, prefix=GLOBAL_RANK_PREFIX):
    return f"{prefix}{tool_id}"


def _top_fraction_mask(series, fraction, *, tie_breaker=None):
    valid = series.notna()
    if not bool(valid.any()):
        return pd.Series(False, index=series.index)
    fraction = float(fraction)
    if fraction <= 0.0:
        return pd.Series(False, index=series.index)
    valid_count = int(valid.sum())
    keep_count = min(valid_count, max(1, int(math.ceil(valid_count * fraction))))
    work = pd.DataFrame({"score": series[valid].astype(float)})
    if tie_breaker is not None:
        work["tie_breaker"] = tie_breaker[valid].astype(str)
        work = work.sort_values(
            ["score", "tie_breaker"], ascending=[False, True], kind="mergesort"
        )
    else:
        work = work.sort_values("score", ascending=False, kind="mergesort")
    selected = pd.Series(False, index=series.index)
    selected.loc[work.index[:keep_count]] = True
    return selected


def _prepare_scored_frame(
    joined, *, score_col, fdr_threshold, abs_logfc_threshold, perturbation=None,
):
    required_cols = {"gene_id", "logFC", "FDR", score_col}
    missing = [col for col in required_cols if col not in joined.columns]
    if missing:
        raise ValueError(f"Joined table missing required columns: {missing}")

    keep_cols = ["gene_id", "logFC", "FDR", score_col]
    for optional in ("dataset_id", "mirna", "perturbation", "PValue"):
        if optional in joined.columns:
            keep_cols.append(optional)

    keep = joined[keep_cols].copy()
    keep = keep[keep["logFC"].notna() & keep["FDR"].notna()].copy()
    keep = keep[keep["FDR"].astype(float) > 0].copy()
    if keep.empty:
        raise ValueError(f"No usable rows remain for {score_col}.")

    total_rows = int(len(keep))
    missing_score_count = int(keep[score_col].isna().sum())
    keep = _annotate_ground_truth(keep, perturbation=perturbation)
    keep["is_positive"] = (
        (keep["FDR"] < fdr_threshold) & (keep["expected_effect"] > abs_logfc_threshold)
    ).astype(int)
    positives_total = int(keep["is_positive"].sum())

    keep = keep[keep[score_col].notna()].copy()
    if keep.empty:
        raise ValueError(f"No scored rows remain for {score_col}.")

    rows_scored = int(len(keep))
    coverage = float(rows_scored / total_rows) if total_rows else float("nan")
    keep[score_col] = keep[score_col].astype(float)
    positives = int(keep["is_positive"].sum())
    negatives = int(len(keep) - positives)
    if positives == 0:
        raise ValueError(f"No positives remain for {score_col}.")
    if negatives == 0:
        raise ValueError(f"No negatives remain for {score_col}.")
    positive_coverage = float(positives / positives_total) if positives_total else float("nan")
    coverage_info = {
        "rows_total": total_rows,
        "rows_scored": rows_scored,
        "rows_missing_score": missing_score_count,
        "coverage": coverage,
        "positives_total": positives_total,
        "positives_scored": positives,
        "positive_coverage": positive_coverage,
    }
    return keep, coverage_info


def _plot_scatter_with_correlation(df, *, score_col, dataset_id, tool_id, out_path):
    pearson = float(df[score_col].corr(df["expected_effect"], method="pearson"))
    spearman = float(df[score_col].corr(df["expected_effect"], method="spearman"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    positive_mask = df["is_positive"].astype(bool)
    negatives = df.loc[~positive_mask]
    positives = df.loc[positive_mask]

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


def _plot_single_predictor_pr_curve(item, *, dataset_id, out_path):
    precision, recall, _ = precision_recall_curve(item["y_true"], item["y_score"])
    pr_auc = auc(recall, precision)
    baseline = float(item["y_true"].mean())

    fig, ax = plt.subplots(figsize=(6.1, 4.9))
    _style_axes(ax, grid_axis="both")
    ax.plot(
        recall,
        precision,
        linewidth=2.2,
        color=CURVE_COLORS[0],
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
            f"  |  positives={int(item['y_true'].sum()):,}"
            f"  |  cov={item['coverage']:.0%}"
        ),
        fontsize=9,
        color=NEUTRAL_COLOR,
    )
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    _save_figure(fig, out_path)


def _plot_predictor_pr_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    _style_axes(ax, grid_axis="both")
    baseline = None
    for index, item in enumerate(comparisons):
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
            color=CURVE_COLORS[index % len(CURVE_COLORS)],
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
    keep["is_positive"] = (
        (keep["FDR"] < fdr_threshold) & (keep["expected_effect"] > abs_logfc_threshold)
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
    for index, item in enumerate(comparisons):
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_score"])
        auroc = roc_auc_score(item["y_true"], item["y_score"])
        ax.plot(
            fpr,
            tpr,
            label=f"{_tool_label(item['tool_id'])} ({auroc:.3f})",
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


def _plot_predictor_gsea_curves(comparisons, *, dataset_id, out_path):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    _style_axes(ax, grid_axis="both")
    for index, item in enumerate(comparisons):
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
            color=CURVE_COLORS[index % len(CURVE_COLORS)],
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


def _plot_algorithms_vs_genes_heatmap(
    joined, *, score_cols, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, perturbation=None,
):
    work = joined[["gene_id", "logFC", "FDR", *score_cols, *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work = _annotate_ground_truth(work, perturbation=perturbation)
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold) & (work["expected_effect"] > abs_logfc_threshold)
    ).astype(int)
    work = work.sort_values(
        ["is_positive", "FDR", "expected_effect"], ascending=[False, True, False],
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
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} genes ordered by benchmark ground truth and expected effect"
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
        label="dataset-local rank percentile",
    )
    _save_figure(fig, out_path)


def _plot_top_positive_heatmap(
    joined, *, rank_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold, positive_fraction, perturbation=None,
):
    work = joined[["gene_id", "logFC", "FDR", *rank_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work = _annotate_ground_truth(work, perturbation=perturbation)
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold) & (work["expected_effect"] > abs_logfc_threshold)
    ).astype(int)
    work = work[work["is_positive"] == 1].copy()
    if work.empty:
        return False

    work = work.sort_values(["FDR", "expected_effect"], ascending=[True, False]).reset_index(drop=True)
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
            f"{_dataset_caption(dataset_id)}  |  {len(work):,} positive genes selected by FDR and expected effect"
            "  |  predictor colors show dataset-local rank percentile"
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
        label="dataset-local rank percentile",
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

    fig, ax = plt.subplots(
        figsize=(max(5.2, len(tool_ids) * 1.6), max(4.6, len(tool_ids) * 1.25))
    )
    image = ax.imshow(matrix.astype(float).to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tool_ids)))
    ax.set_xticklabels([_tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")
    ax.set_yticks(range(len(tool_ids)))
    ax.set_yticklabels([_tool_label(tool_id) for tool_id in tool_ids])
    ax.set_title(
        f"Predictor agreement (shared local top {int(top_fraction * 100)}%)",
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
    scatter_png, pr_curve_png, gsea_png,
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
        (
            f"- GT positives: `FDR < {fdr_threshold}` and perturbation-aware effect `> {abs_logfc_threshold}`"
            " (`-logFC` for OE, `+logFC` for KO/KD)"
        ),
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
    gsea_png=None,
):
    with PdfPages(pdf_path) as pdf:
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
                (
                    f"GT positives: FDR < {fdr_threshold} and perturbation-aware effect > {abs_logfc_threshold} "
                    "(-logFC for OE, +logFC for KO/KD)"
                ),
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
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        plot_specs = [
            ("Score vs expected effect", scatter_png),
            ("Precision-recall curve", pr_curve_png),
            ("GSEA enrichment", gsea_png),
        ]
        existing_plots = [(label, path) for label, path in plot_specs if path and pathlib.Path(path).is_file()]
        if existing_plots:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            fig.patch.set_facecolor("white")
            ax.text(
                0.06,
                0.975,
                f"{dataset_id} | {_tool_label(tool_id)} | Plots",
                fontsize=13,
                fontweight="bold",
                color="#17324D",
                va="top",
                ha="left",
            )
            ax.text(
                0.06,
                0.945,
                "These are the main per-tool visuals for this dataset: ranking quality, classification quality, and enrichment behavior.",
                fontsize=9.2,
                color="#22303C",
                va="top",
                ha="left",
                wrap=True,
            )
            layout_specs = [
                ("Score vs expected effect", [0.08, 0.55, 0.84, 0.28]),
                ("Precision-recall curve", [0.08, 0.12, 0.40, 0.25]),
                ("GSEA enrichment", [0.52, 0.12, 0.40, 0.25]),
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
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _write_tool_report(
    *, dataset_id, mirna, cell_line, perturbation, geo_accession,
    de_table_path, joined_tsv,
    tool_id, predictor_output_path, metrics, markdown_path, pdf_path, coverage_info,
    scatter_png, pr_curve_png, gsea_png, fdr_threshold, abs_logfc_threshold,
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
        gsea_png=gsea_png,
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
    dataset_plots_dir = plots_dir
    predictor_plots_dir = dataset_plots_dir / "predictors"
    comparison_plots_dir = dataset_plots_dir / "comparisons"
    heatmap_plots_dir = dataset_plots_dir / "heatmaps"
    for path in (dataset_plots_dir, predictor_plots_dir, comparison_plots_dir, heatmap_plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    tool_ids = [_tool_id_from_score_col(sc) for sc in score_cols]
    global_rank_cols = []
    local_rank_cols = []
    for score_col, tool_id in zip(score_cols, tool_ids):
        global_rank_col = _rank_col_for_tool(tool_id)
        local_rank_col = _rank_col_for_tool(tool_id, prefix=LOCAL_RANK_PREFIX)
        if global_rank_col not in joined.columns or local_rank_col not in joined.columns:
            joined = joined.copy()
        if local_rank_col not in joined.columns:
            joined[local_rank_col] = _rank_scale_scores(joined[score_col])
        if global_rank_col not in joined.columns:
            joined[global_rank_col] = joined[local_rank_col]
        global_rank_cols.append(global_rank_col)
        local_rank_cols.append(local_rank_col)

    metric_rows = []
    dataset_plots = {}
    predictor_correlation_tsv = None
    comparisons = []
    coverage_by_tool = {}

    _emit_log(logger, f"    Evaluation start: {dataset_id} | tools={tool_ids}")

    for score_col, tool_id in zip(score_cols, tool_ids):
        _emit_log(logger, f"    Tool: {tool_id} | preparing scored pairs")
        tool_plots_dir = predictor_plots_dir / tool_id
        tool_plots_dir.mkdir(parents=True, exist_ok=True)
        scored, coverage_info = _prepare_scored_frame(
            joined, score_col=score_col,
            fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
            perturbation=perturbation,
        )
        scatter_png = tool_plots_dir / "score_vs_expected_effect.png"
        gsea_png = tool_plots_dir / "gsea_enrichment.png"
        pr_curve_png = tool_plots_dir / "precision_recall_curve.png"
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
        _plot_single_predictor_pr_curve(
            {
                "tool_id": tool_id,
                "y_true": scored["is_positive"],
                "y_score": scored[score_col],
                "coverage": coverage_info["coverage"],
            },
            dataset_id=dataset_id,
            out_path=pr_curve_png,
        )

        report_md = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.md"
        report_pdf = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.pdf"
        metrics = {
            "rows_total": float(coverage_info["rows_total"]),
            "rows_used": float(len(scored)),
            "positives": float(scored["is_positive"].sum()),
            "negatives": float(len(scored) - int(scored["is_positive"].sum())),
            "rows_missing_score": float(coverage_info["rows_missing_score"]),
            "coverage": float(coverage_info["coverage"]),
            "positives_total": float(coverage_info["positives_total"]),
            "positives_scored": float(coverage_info["positives_scored"]),
            "positive_coverage": float(coverage_info["positive_coverage"]),
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
            coverage_info=coverage_info,
            scatter_png=scatter_png,
            pr_curve_png=pr_curve_png,
            gsea_png=gsea_png,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
        )
        _emit_log(
            logger,
            (
                f"    Tool: {tool_id} | coverage={coverage_info['coverage']:.1%} "
                f"| positive_cov={coverage_info['positive_coverage']:.1%} "
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
            "positive_coverage": coverage_info["positive_coverage"],
            "aps": aps, "spearman": spearman, "auroc": auroc, "pr_auc": pr_auc,
        })
        comparisons.append({
            "tool_id": tool_id,
            "y_true": scored["is_positive"],
            "y_score": scored[score_col],
            "coverage": coverage_info["coverage"],
        })
        coverage_by_tool[tool_id] = coverage_info["coverage"]
        dataset_plots[f"{tool_id}_scatter"] = str(scatter_png)
        dataset_plots[f"{tool_id}_gsea_enrichment"] = str(gsea_png)
        dataset_plots[f"{tool_id}_pr_curve"] = str(pr_curve_png)

    heatmap_png = heatmap_plots_dir / "algorithms_vs_genes.png"
    _plot_algorithms_vs_genes_heatmap(
        joined, score_cols=score_cols, rank_cols=local_rank_cols, tool_ids=tool_ids,
        dataset_id=dataset_id, out_path=heatmap_png,
        fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
        perturbation=perturbation,
    )
    dataset_plots["algorithms_vs_genes_heatmap"] = str(heatmap_png)
    _emit_log(logger, f"    Dataset: {dataset_id} | wrote gene-level heatmap")

    top_positive_heatmap_png = heatmap_plots_dir / "top_10pct_positive_genes.png"
    wrote_top_positive_heatmap = _plot_top_positive_heatmap(
        joined,
        rank_cols=local_rank_cols,
        tool_ids=tool_ids,
        dataset_id=dataset_id,
        out_path=top_positive_heatmap_png,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        positive_fraction=0.10,
        perturbation=perturbation,
    )
    if wrote_top_positive_heatmap:
        dataset_plots["top_10pct_positive_heatmap"] = str(top_positive_heatmap_png)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote top-positive heatmap")

    if len(score_cols) >= 2:
        comparison_pr_png = comparison_plots_dir / "precision_recall_common.png"
        comparison_roc_png = comparison_plots_dir / "roc_common.png"
        comparison_gsea_png = comparison_plots_dir / "gsea_common.png"
        common_pr = _prepare_common_scored_frame(
            joined,
            score_cols=score_cols,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            perturbation=perturbation,
        )
        common_comparisons = [
            {
                "tool_id": tool_id,
                "gene_id": common_pr["gene_id"],
                "y_true": common_pr["is_positive"],
                "y_score": common_pr[score_col],
                "coverage": coverage_by_tool.get(tool_id, float("nan")),
            }
            for score_col, tool_id in zip(score_cols, tool_ids)
        ]
        _plot_predictor_pr_curves(
            common_comparisons,
            dataset_id=dataset_id,
            out_path=comparison_pr_png,
        )
        _plot_predictor_roc_curves(
            common_comparisons,
            dataset_id=dataset_id,
            out_path=comparison_roc_png,
        )
        _plot_predictor_gsea_curves(
            common_comparisons,
            dataset_id=dataset_id,
            out_path=comparison_gsea_png,
        )
        dataset_plots["predictor_pr_curves"] = str(comparison_pr_png)
        dataset_plots["predictor_roc_curves"] = str(comparison_roc_png)
        dataset_plots["predictor_gsea_curves"] = str(comparison_gsea_png)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote PR/ROC/GSEA comparison plots")

        corr_png = comparison_plots_dir / "predictor_correlation_heatmap.png"
        corr_tsv = reports_dir / f"{dataset_id}__predictor_correlation.tsv"
        corr_matrix = _plot_predictor_correlation_heatmap(
            joined, rank_cols=local_rank_cols, tool_ids=tool_ids,
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
        ("positive_coverage", "positive_coverage_per_experiment.tsv"),
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
    if float(np.nanmin(matrix)) < 0.0:
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
            aspect="auto",
        )
    else:
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
    for ax, metric_name in zip(axes, metric_names):
        ymin, ymax = _metric_plot_limits(metric_name)
        ax.set_ylim(ymin, ymax)
    _save_figure(fig, out_path)


def _plot_metric_vs_performance(summary_df, *, x_metric_col, x_label, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=False)
    metric_specs = [("aps_mean", "Mean APS"), ("auroc_mean", "Mean AUROC")]
    x_values = summary_df[x_metric_col].astype(float).to_numpy()

    for ax, (metric_col, metric_label) in zip(axes, metric_specs):
        _style_axes(ax, grid_axis="both")
        values = summary_df[metric_col].astype(float).to_numpy()
        for index, tool_id in enumerate(summary_df["tool_id"].tolist()):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            ax.scatter(
                x_values[index],
                values[index],
                s=80,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.text(
                x_values[index] + 0.01,
                values[index],
                _tool_label(tool_id),
                fontsize=8.5,
                color="#22303C",
                va="center",
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
    axes[0].set_title(
        title,
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    _save_figure(fig, out_path)


def _plot_coverage_vs_performance(summary_df, *, out_path):
    _plot_metric_vs_performance(
        summary_df,
        x_metric_col="coverage_mean",
        x_label="Mean coverage",
        title="Coverage vs performance",
        out_path=out_path,
    )


def _plot_positive_coverage_vs_performance(summary_df, *, out_path):
    _plot_metric_vs_performance(
        summary_df,
        x_metric_col="positive_coverage_mean",
        x_label="Mean positive coverage",
        title="Positive coverage vs performance",
        out_path=out_path,
    )


def _plot_rank_class_distributions(joined_frames, *, out_path, fdr_threshold, abs_logfc_threshold):
    combined = pd.concat(joined_frames, ignore_index=True)
    rank_cols = sorted(col for col in combined.columns if col.startswith(GLOBAL_RANK_PREFIX))
    if not rank_cols:
        return False

    keep_cols = ["logFC", "FDR", *rank_cols]
    for optional in ("dataset_id", "perturbation"):
        if optional in combined.columns:
            keep_cols.append(optional)
    work = combined[keep_cols].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    if work.empty:
        return False
    work = _annotate_ground_truth(work)
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold) & (work["expected_effect"] > abs_logfc_threshold)
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
    metric_plots_dir = plots_dir / "metrics"
    coverage_plots_dir = plots_dir / "coverage"
    rank_plots_dir = plots_dir / "ranks"
    for path in (metric_plots_dir, coverage_plots_dir, rank_plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        return {
            "tables": {},
            "plots": {},
        }

    metric_names = ["coverage", "positive_coverage", "aps", "pr_auc", "spearman", "auroc"]
    summary = metrics_df.groupby("tool_id")[metric_names].agg(["count", "mean", "median", "std", "min", "max"])
    summary.columns = [f"{metric_name}_{stat_name}" for metric_name, stat_name in summary.columns]
    summary = summary.reset_index()
    summary_path = tables_dir / "cross_dataset_predictor_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    _emit_log(logger, f"  Wrote cross-dataset summary table: {summary_path}")

    distributions_path = metric_plots_dir / "cross_dataset_metric_distributions.png"
    _plot_cross_dataset_metric_distributions(metrics_df, metric_names=metric_names, out_path=distributions_path)
    _emit_log(logger, f"  Wrote cross-dataset metric distributions: {distributions_path}")

    positive_coverage_scatter_path = coverage_plots_dir / "positive_coverage_vs_performance.png"
    _plot_positive_coverage_vs_performance(summary, out_path=positive_coverage_scatter_path)
    _emit_log(
        logger,
        f"  Wrote positive coverage vs performance plot: {positive_coverage_scatter_path}",
    )

    rank_distribution_path = rank_plots_dir / "positive_background_rank_distributions.png"
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
            "cross_dataset_metric_distributions": str(distributions_path),
            "positive_coverage_vs_performance": str(positive_coverage_scatter_path),
            **(
                {"positive_background_rank_distributions": str(rank_distribution_path)}
                if wrote_rank_distributions
                else {}
            ),
        },
    }
