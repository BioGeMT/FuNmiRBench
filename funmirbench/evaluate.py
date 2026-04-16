"""Evaluate joined GT/prediction tables: metrics, plots, reports."""

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

SCORE_PREFIX = "score_"
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


def _normalize_scores(series):
    values = series.astype(float)
    lo, hi = values.min(skipna=True), values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(float("nan"), index=series.index)
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo)


def _tool_id_from_score_col(score_col):
    if score_col.startswith(SCORE_PREFIX):
        return score_col[len(SCORE_PREFIX):]
    return score_col


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
    joined, *, score_cols, tool_ids, dataset_id, out_path,
    fdr_threshold, abs_logfc_threshold,
):
    work = joined[["gene_id", "logFC", "FDR", *score_cols]].copy()
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

    normalized = pd.DataFrame({
        sc: _normalize_scores(work[sc]) for sc in score_cols
    })

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
        np.ma.masked_invalid(normalized.to_numpy(dtype=float)),
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
        label="normalized score",
    )
    _save_figure(fig, out_path)


def _plot_predictor_correlation_heatmap(
    joined, *, score_cols, tool_ids, dataset_id, out_path, top_fraction,
):
    normalized = {
        tid: _normalize_scores(joined[sc])
        for tid, sc in zip(tool_ids, score_cols)
    }
    top_masks = {tid: _top_fraction_mask(normalized[tid], top_fraction) for tid in tool_ids}

    matrix = pd.DataFrame(index=tool_ids, columns=tool_ids, dtype=float)
    for a in tool_ids:
        for b in tool_ids:
            pair_mask = normalized[a].notna() & normalized[b].notna()
            union_mask = pair_mask & (top_masks[a] | top_masks[b])
            if int(union_mask.sum()) < 2:
                corr = 1.0 if a == b else float("nan")
            else:
                corr = float(normalized[a][union_mask].corr(normalized[b][union_mask], method="spearman"))
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


def _write_tool_report(
    *, dataset_id, mirna, cell_line, perturbation, geo_accession,
    de_table_path, joined_tsv,
    tool_id, predictor_output_path, metrics, out_path, coverage_info, scatter_png,
):
    lines = [
        f"dataset_id: {dataset_id}",
        f"mirna: {mirna or 'NA'}",
        f"cell_line: {cell_line or 'NA'}",
        f"perturbation: {perturbation or 'NA'}",
        f"geo_accession: {geo_accession or 'NA'}",
        f"de_table_path: {de_table_path or 'NA'}",
        f"joined_tsv: {joined_tsv or 'NA'}",
        f"tool_id: {tool_id}",
        f"predictor_output_path: {predictor_output_path or 'NA'}",
        f"rows_total: {int(coverage_info['rows_total'])}",
        f"rows_scored: {int(coverage_info['rows_scored'])}",
        f"rows_missing_score: {int(coverage_info['rows_missing_score'])}",
        f"coverage: {coverage_info['coverage']:.6f}",
        "",
        "metrics:",
        f"  rows_used: {int(metrics['rows_used'])}",
        f"  positives: {int(metrics['positives'])}",
        f"  negatives: {int(metrics['negatives'])}",
        f"  pearson: {metrics['pearson']:.6f}",
        f"  spearman: {metrics['spearman']:.6f}",
        f"  aps: {metrics['aps']:.6f}",
        f"  pr_auc: {metrics['pr_auc']:.6f}",
        f"  auroc: {metrics['auroc']:.6f}",
        "",
        "plots:",
        f"  scatter: {scatter_png}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        pearson, spearman = _plot_scatter_with_correlation(
            scored,
            score_col=score_col,
            dataset_id=dataset_id,
            tool_id=tool_id,
            out_path=scatter_png,
        )
        pr_auc, aps = _compute_pr_metrics(scored["is_positive"], scored[score_col])
        auroc = _compute_auroc(scored["is_positive"], scored[score_col])

        report_txt = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.txt"
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
            metrics=metrics, out_path=report_txt,
            coverage_info=coverage_info, scatter_png=scatter_png,
        )
        _emit_log(
            logger,
            (
                f"    Tool: {tool_id} | coverage={coverage_info['coverage']:.1%} "
                f"| rows={coverage_info['rows_scored']}/{coverage_info['rows_total']} "
                f"| APS={aps:.3f} | AUROC={auroc:.3f}"
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

    heatmap_png = dataset_plots_dir / "algorithms_vs_genes_heatmap.png"
    _plot_algorithms_vs_genes_heatmap(
        joined, score_cols=score_cols, tool_ids=tool_ids,
        dataset_id=dataset_id, out_path=heatmap_png,
        fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
    )
    dataset_plots["algorithms_vs_genes_heatmap"] = str(heatmap_png)
    _emit_log(logger, f"    Dataset: {dataset_id} | wrote gene-level heatmap")

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
            joined, score_cols=score_cols, tool_ids=tool_ids,
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
