"""Predictor-combination analysis for benchmark reports.

This module tests whether combining real predictors improves performance over the
best single predictor. The primary output is a coverage-aware frontier plot:
combinations are only useful if they improve performance without sacrificing too
much sensitivity/coverage.
"""

from __future__ import annotations

import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score

from funmirbench.evaluate import (
    FIGURE_DPI,
    GRID_COLOR,
    NEUTRAL_COLOR,
    SCORE_PREFIX,
    _annotate_ground_truth,
    _positive_mask,
    _rank_scale_scores,
    _save_figure,
    _style_axes,
)


EXCLUDED_COMBINATION_TOOL_IDS = {
    "random",
    "random_3000",
    "cheating",
    "perfect",
}
DEFAULT_MIN_DATASET_COVERAGE = 0.01
DEFAULT_MAX_COMBINATION_SIZE = 3


def _score_col(tool_id: str) -> str:
    return f"{SCORE_PREFIX}{tool_id}"


def _eligible_real_tools(tool_ids, joined_frames, *, excluded_tool_ids=None, min_dataset_coverage=DEFAULT_MIN_DATASET_COVERAGE):
    excluded_tool_ids = set(excluded_tool_ids or EXCLUDED_COMBINATION_TOOL_IDS)
    eligible = []
    for tool_id in tool_ids:
        tool_id = str(tool_id)
        if tool_id in excluded_tool_ids:
            continue
        col = _score_col(tool_id)
        coverage_values = []
        for joined in joined_frames:
            if col not in joined.columns:
                continue
            coverage_values.append(float(joined[col].notna().mean()))
        if coverage_values and max(coverage_values) >= float(min_dataset_coverage):
            eligible.append(tool_id)
    return eligible


def _iter_tool_combinations(tool_ids, *, max_combination_size=DEFAULT_MAX_COMBINATION_SIZE):
    max_size = min(int(max_combination_size), len(tool_ids))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(tool_ids, size):
            yield tuple(combo)


def _prepare_combo_frame(joined, combo, *, fdr_threshold, abs_logfc_threshold):
    score_cols = [_score_col(tool_id) for tool_id in combo]
    required = {"gene_id", "logFC", "FDR", *score_cols}
    missing = [col for col in required if col not in joined.columns]
    if missing:
        return None
    keep_cols = ["gene_id", "logFC", "FDR", *score_cols]
    for optional in ("dataset_id", "mirna", "perturbation", "PValue"):
        if optional in joined.columns:
            keep_cols.append(optional)
    work = joined[keep_cols].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work = work[work["FDR"].astype(float) > 0].copy()
    if work.empty:
        return None
    work = _annotate_ground_truth(work)
    work["is_positive"] = _positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
    positives_total = int(work["is_positive"].sum())
    total_rows = int(len(work))
    if positives_total == 0 or total_rows == 0:
        return None

    rank_cols = []
    for tool_id, score_col in zip(combo, score_cols):
        rank_col = f"rank_{tool_id}"
        work[rank_col] = _rank_scale_scores(work[score_col])
        rank_cols.append(rank_col)
    work["combo_score"] = work[rank_cols].mean(axis=1, skipna=True)
    work = work[work["combo_score"].notna()].copy()
    if work.empty:
        return None
    positives_scored = int(work["is_positive"].sum())
    negatives_scored = int(len(work) - positives_scored)
    if positives_scored == 0 or negatives_scored == 0:
        return None
    return work, {
        "rows_total": total_rows,
        "rows_scored": int(len(work)),
        "coverage": float(len(work) / total_rows),
        "positives_total": positives_total,
        "positives_scored": positives_scored,
        "positive_coverage": float(positives_scored / positives_total),
    }


def _evaluate_combo_dataset(joined, combo, *, fdr_threshold, abs_logfc_threshold):
    prepared = _prepare_combo_frame(
        joined,
        combo,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    )
    if prepared is None:
        return None
    work, coverage_info = prepared
    y_true = work["is_positive"].astype(int).to_numpy()
    y_score = work["combo_score"].astype(float).to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    top_n = min(100, len(work))
    top = work.sort_values(["combo_score", "gene_id"], ascending=[False, True], kind="mergesort").head(top_n)
    precision_at_top_n = float(top["is_positive"].mean()) if top_n else float("nan")
    return {
        **coverage_info,
        "aps": float(average_precision_score(y_true, y_score)),
        "pr_auc": float(auc(recall, precision)),
        "auroc": auroc,
        "precision_at_top_100": precision_at_top_n,
        "top_n": int(top_n),
    }


def _summarize_combo_rows(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    summary = {}
    for metric in [
        "coverage",
        "positive_coverage",
        "aps",
        "pr_auc",
        "auroc",
        "precision_at_top_100",
    ]:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        summary[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
        summary[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
        summary[f"{metric}_count"] = int(values.count())
    return summary


def compute_predictor_combination_summary(
    joined_frames,
    *,
    tool_ids,
    fdr_threshold,
    abs_logfc_threshold,
    max_combination_size=DEFAULT_MAX_COMBINATION_SIZE,
    excluded_tool_ids=None,
):
    real_tools = _eligible_real_tools(tool_ids, joined_frames, excluded_tool_ids=excluded_tool_ids)
    rows = []
    for combo in _iter_tool_combinations(real_tools, max_combination_size=max_combination_size):
        dataset_rows = []
        for joined in joined_frames:
            result = _evaluate_combo_dataset(
                joined,
                combo,
                fdr_threshold=fdr_threshold,
                abs_logfc_threshold=abs_logfc_threshold,
            )
            if result is not None:
                dataset_rows.append(result)
        summary = _summarize_combo_rows(dataset_rows)
        if summary is None:
            continue
        rows.append(
            {
                "combination_id": "+".join(combo),
                "tool_ids": ",".join(combo),
                "combination_size": len(combo),
                "ensemble_rule": "rank_mean_available",
                "dataset_count": len(dataset_rows),
                **summary,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["aps_mean", "positive_coverage_mean", "coverage_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _pareto_frontier_mask(df, *, x_col="positive_coverage_mean", y_col="aps_mean"):
    values = df[[x_col, y_col]].astype(float)
    mask = []
    for idx, row in values.iterrows():
        dominated = False
        for jdx, other in values.iterrows():
            if idx == jdx:
                continue
            if (
                other[x_col] >= row[x_col]
                and other[y_col] >= row[y_col]
                and (other[x_col] > row[x_col] or other[y_col] > row[y_col])
            ):
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)


def write_predictor_combination_frontier_plot(summary_df, out_path, *, title="Predictor-combination performance frontier"):
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    _style_axes(ax, grid_axis="both")
    if summary_df.empty:
        ax.text(0.5, 0.5, "No real predictor combinations available", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path

    work = summary_df.copy()
    work["is_frontier"] = _pareto_frontier_mask(work)
    markers = {1: "o", 2: "s", 3: "^"}
    labels = {1: "single predictor", 2: "pair", 3: "triple"}
    for size in sorted(work["combination_size"].unique()):
        subset = work[work["combination_size"] == size]
        ax.scatter(
            subset["positive_coverage_mean"],
            subset["aps_mean"],
            s=np.where(subset["is_frontier"], 92, 48),
            marker=markers.get(int(size), "D"),
            alpha=np.where(subset["is_frontier"], 0.95, 0.45),
            edgecolor="black",
            linewidth=0.65,
            label=labels.get(int(size), f"size {size}"),
        )
    frontier = work[work["is_frontier"]].sort_values("positive_coverage_mean")
    if len(frontier) > 1:
        ax.plot(frontier["positive_coverage_mean"], frontier["aps_mean"], color=NEUTRAL_COLOR, linewidth=1.4, linestyle="--", label="Pareto frontier")
    best_single = work[work["combination_size"] == 1].sort_values("aps_mean", ascending=False).head(1)
    label_rows = pd.concat([frontier, best_single], ignore_index=True).drop_duplicates("combination_id")
    for _, row in label_rows.iterrows():
        ax.annotate(
            str(row["combination_id"]).replace("+", " + "),
            (row["positive_coverage_mean"], row["aps_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.2,
        )
    ax.set_xlabel("Mean positive coverage", fontsize=10)
    ax.set_ylabel("Mean APS", fontsize=10)
    ax.set_xlim(0, min(1.02, max(0.15, float(work["positive_coverage_mean"].max()) * 1.18)))
    ax.set_ylim(0, min(1.02, max(0.2, float(work["aps_mean"].max()) * 1.22)))
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=14)
    fig.text(
        0.125,
        0.94,
        "Rank-mean ensembles over available scores; random/oracle predictors excluded. Points on the frontier are not dominated in both coverage and APS.",
        fontsize=8.8,
        color=NEUTRAL_COLOR,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _save_figure(fig, out_path)
    return out_path


def write_predictor_combination_outputs(
    joined_frames,
    out_tables_dir,
    out_plots_dir,
    *,
    tool_ids,
    fdr_threshold,
    abs_logfc_threshold,
    max_combination_size=DEFAULT_MAX_COMBINATION_SIZE,
    logger=None,
):
    out_tables_dir = pathlib.Path(out_tables_dir)
    out_plots_dir = pathlib.Path(out_plots_dir) / "combinations"
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    summary_df = compute_predictor_combination_summary(
        joined_frames,
        tool_ids=tool_ids,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        max_combination_size=max_combination_size,
    )
    table_path = out_tables_dir / "predictor_combination_summary.tsv"
    summary_df.to_csv(table_path, sep="\t", index=False)
    plot_path = out_plots_dir / "predictor_combination_frontier.png"
    write_predictor_combination_frontier_plot(summary_df, plot_path)
    if logger is not None:
        logger(f"Wrote predictor-combination summary: {table_path}")
        logger(f"Wrote predictor-combination frontier: {plot_path}")
    return {
        "tables": {"predictor_combination_summary": str(table_path)},
        "plots": {"predictor_combination_frontier": str(plot_path)},
    }
