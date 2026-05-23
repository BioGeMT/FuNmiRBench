"""Predictor-combination analysis for benchmark reports.

This module tests whether combining predictors improves over individual
predictors. It writes the coverage-versus-APS frontier and the supporting
combination summary table.
"""

from __future__ import annotations

import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from funmirbench.evaluate import (
    NEUTRAL_COLOR,
    PLOT_AXIS_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    SCORE_PREFIX,
    _add_figure_heading,
    _annotate_ground_truth,
    _positive_mask,
    _rank_scale_scores,
    _save_figure,
    _style_axes,
    _tool_label,
    _top_fraction_mask,
)


EXCLUDED_COMBINATION_TOOL_IDS = {
    "random",
    "random_3000",
    "cheating",
    "perfect",
}
DEFAULT_MIN_DATASET_COVERAGE = 0.01
DEFAULT_MAX_COMBINATION_SIZE = None
DEFAULT_PREDICTOR_TOP_FRACTION = 0.10
SINGLE_PREDICTOR_COLOR = "#111827"
ORIGINAL_SINGLE_PREDICTOR_COLOR = "#1F77B4"
ORIGINAL_COMBINATION_COLOR = "#FF7F0E"
COMBINATION_CMAP = "plasma"


def _score_col(tool_id: str) -> str:
    return f"{SCORE_PREFIX}{tool_id}"


def _eligible_real_tools(
    tool_ids,
    joined_frames,
    *,
    excluded_tool_ids=None,
    min_dataset_coverage=DEFAULT_MIN_DATASET_COVERAGE,
):
    excluded_tool_ids = set(
        EXCLUDED_COMBINATION_TOOL_IDS if excluded_tool_ids is None else excluded_tool_ids
    )
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
    max_size = (
        len(tool_ids)
        if max_combination_size is None
        else min(int(max_combination_size), len(tool_ids))
    )
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


def _evaluate_combo_dataset(
    joined,
    combo,
    *,
    fdr_threshold,
    abs_logfc_threshold,
    predictor_top_fraction,
):
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
    intersection_info = _evaluate_top_fraction_intersection_dataset(
        joined,
        combo,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
    )
    return {
        **coverage_info,
        "aps": float(average_precision_score(y_true, y_score)),
        "pr_auc": float(auc(recall, precision)),
        "auroc": auroc,
        "precision_at_top_100": precision_at_top_n,
        "top_n": int(top_n),
        **(intersection_info or {}),
    }


def _evaluate_top_fraction_intersection_dataset(
    joined,
    combo,
    *,
    fdr_threshold,
    abs_logfc_threshold,
    predictor_top_fraction,
):
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
    if positives_total <= 0:
        return None

    tie_breaker = work["gene_id"] if "gene_id" in work.columns else None
    selected = pd.Series(True, index=work.index)
    for score_col in score_cols:
        ranks = _rank_scale_scores(work[score_col])
        selected &= _top_fraction_mask(
            ranks,
            predictor_top_fraction,
            tie_breaker=tie_breaker,
        )

    selected_positive = int((selected & (work["is_positive"] == 1)).sum())
    selected_background = int((selected & (work["is_positive"] == 0)).sum())
    selected_total = selected_positive + selected_background
    return {
        "intersection_top_fraction": float(predictor_top_fraction),
        "intersection_selected": selected_total,
        "intersection_positives": selected_positive,
        "intersection_background": selected_background,
        "intersection_precision": (
            float(selected_positive / selected_total)
            if selected_total
            else float("nan")
        ),
        "intersection_recovery": float(selected_positive / positives_total),
        "intersection_coverage": float(selected_total / len(work)),
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
        "intersection_selected",
        "intersection_positives",
        "intersection_background",
        "intersection_precision",
        "intersection_recovery",
        "intersection_coverage",
    ]:
        if metric not in df.columns:
            summary[f"{metric}_mean"] = float("nan")
            summary[f"{metric}_median"] = float("nan")
            summary[f"{metric}_count"] = 0
            continue
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
    predictor_top_fraction=DEFAULT_PREDICTOR_TOP_FRACTION,
    excluded_tool_ids=None,
    min_dataset_coverage=DEFAULT_MIN_DATASET_COVERAGE,
):
    real_tools = _eligible_real_tools(
        tool_ids,
        joined_frames,
        excluded_tool_ids=excluded_tool_ids,
        min_dataset_coverage=min_dataset_coverage,
    )
    rows = []
    for combo in _iter_tool_combinations(real_tools, max_combination_size=max_combination_size):
        dataset_rows = []
        for joined in joined_frames:
            result = _evaluate_combo_dataset(
                joined,
                combo,
                fdr_threshold=fdr_threshold,
                abs_logfc_threshold=abs_logfc_threshold,
                predictor_top_fraction=predictor_top_fraction,
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
                "contains_control_predictor": any(
                    tool_id in EXCLUDED_COMBINATION_TOOL_IDS for tool_id in combo
                ),
                "ensemble_rule": "rank_mean_available",
                "intersection_rule": f"top_{float(predictor_top_fraction):.0%}_intersection",
                "dataset_count": len(dataset_rows),
                **summary,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    single_rows = out[out["combination_size"] == 1]
    single_reference_rows = single_rows[~single_rows["contains_control_predictor"]]
    if single_reference_rows.empty:
        single_reference_rows = single_rows
    best_single_aps = single_reference_rows["aps_mean"].max()
    out["delta_aps_vs_best_single"] = out["aps_mean"] - best_single_aps
    out["beats_best_single_aps"] = out["delta_aps_vs_best_single"] > 0
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


def _short_tool_label(tool_id):
    label = _tool_label(tool_id)
    replacements = {
        "TargetScan v8": "TargetScan",
        "Random (3000 per dataset)": "Random 3000",
        "mirdb mirtarget": "miRDB",
    }
    return replacements.get(label, label.replace("_", " "))


def _combination_label(combination_id):
    return " + ".join(_short_tool_label(part) for part in str(combination_id).split("+"))


def _has_control_predictor(tool_ids):
    return any(
        tool_id in EXCLUDED_COMBINATION_TOOL_IDS
        for tool_id in str(tool_ids).split(",")
    )


def _combination_size_color(size, *, min_size, max_size):
    cmap = plt.get_cmap(COMBINATION_CMAP)
    if max_size <= min_size:
        return cmap(0.72)
    fraction = (int(size) - min_size) / float(max_size - min_size)
    return cmap(0.18 + 0.70 * fraction)


def _expanded_frontier_label_position(combination_id):
    positions = {
        "tec-mitarget": (0.035, 0.185),
        "targetscan": (0.286, 0.154),
        "targetscan+tec-mitarget": (0.105, 0.116),
        "mirdb_mirtarget": (0.308, 0.132),
        "mirdb_mirtarget+tec-mitarget": (0.245, 0.091),
        "targetscan+mirdb_mirtarget": (0.410, 0.130),
        "targetscan+mirdb_mirtarget+tec-mitarget": (0.408, 0.082),
    }
    return positions.get(str(combination_id))


def write_predictor_combination_frontier_plot(
    summary_df,
    out_path,
    *,
    title="Predictor-combination performance frontier",
):
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    _style_axes(ax, grid_axis="both")
    if summary_df.empty:
        ax.text(0.5, 0.5, "No predictor combinations available", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path

    work = summary_df.copy()
    if work.empty:
        ax.text(0.5, 0.5, "No predictor combinations available", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path

    work["is_frontier"] = _pareto_frontier_mask(work)
    markers = {1: "o", 2: "s", 3: "^"}
    labels = {1: "single predictor", 2: "pair", 3: "triple"}
    for size in sorted(work["combination_size"].unique()):
        subset = work[work["combination_size"] == size]
        ax.scatter(
            subset["positive_coverage_mean"],
            subset["aps_mean"],
            s=np.where(subset["is_frontier"], 120, 62),
            marker=markers.get(int(size), "D"),
            color=ORIGINAL_SINGLE_PREDICTOR_COLOR if int(size) == 1 else ORIGINAL_COMBINATION_COLOR,
            alpha=np.where(subset["is_frontier"], 0.95, 0.45),
            edgecolor="black",
            linewidth=0.85,
            label=labels.get(int(size), f"size {size}"),
        )
    frontier = work[work["is_frontier"]].sort_values("positive_coverage_mean")
    if len(frontier) > 1:
        ax.plot(
            frontier["positive_coverage_mean"],
            frontier["aps_mean"],
            color=NEUTRAL_COLOR,
            linewidth=1.9,
            linestyle="--",
            label="Pareto frontier",
        )

    best_single = work[work["combination_size"] == 1].sort_values("aps_mean", ascending=False).head(1)
    label_rows = pd.concat([frontier, best_single], ignore_index=True).drop_duplicates("combination_id")
    for _, row in label_rows.iterrows():
        ax.annotate(
            str(row["combination_id"]).replace("+", " + "),
            (row["positive_coverage_mean"], row["aps_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10.0,
        )
    ax.set_xlabel("Mean positive coverage", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean APS", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_xlim(0, min(1.02, max(0.15, float(work["positive_coverage_mean"].max()) * 1.18)))
    ax.set_ylim(0, min(1.02, max(0.2, float(work["aps_mean"].max()) * 1.22)))
    _add_figure_heading(
        fig,
        title=title,
        subtitle="Points on the frontier are not dominated in both positive coverage and APS.",
    )
    ax.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc="lower right")
    _save_figure(fig, out_path)
    return out_path


def write_predictor_combination_expanded_frontier_plot(
    summary_df,
    out_path,
    *,
    title="Predictor-combination expanded performance frontier",
):
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    _style_axes(ax, grid_axis="both")
    if summary_df.empty:
        ax.text(0.5, 0.5, "No predictor combinations available", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path

    work = summary_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["positive_coverage_mean", "aps_mean"]
    ).copy()
    if work.empty:
        ax.text(0.5, 0.5, "No predictor combinations available", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path

    if "contains_control_predictor" not in work.columns:
        work["contains_control_predictor"] = work["tool_ids"].map(_has_control_predictor)
    work = work[~work["contains_control_predictor"].astype(bool)].copy()
    if work.empty:
        ax.text(0.5, 0.5, "No predictor combinations available", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path

    work["is_frontier"] = _pareto_frontier_mask(work)
    max_combination_size = int(work["combination_size"].max())
    min_combination_size = (
        int(max(2, work.loc[work["combination_size"] > 1, "combination_size"].min()))
        if (work["combination_size"] > 1).any()
        else 2
    )
    for size in sorted(work["combination_size"].unique()):
        subset = work[work["combination_size"] == size]
        is_single = int(size) == 1
        color = (
            SINGLE_PREDICTOR_COLOR
            if is_single
            else _combination_size_color(
                size,
                min_size=min_combination_size,
                max_size=max_combination_size,
            )
        )
        ax.scatter(
            subset["positive_coverage_mean"],
            subset["aps_mean"],
            s=np.where(subset["is_frontier"], 92, 48),
            marker="o" if is_single else "s",
            color=color,
            alpha=np.where(subset["is_frontier"], 0.95, 0.55),
            edgecolor="black",
            linewidth=0.65,
            zorder=3,
        )

    frontier = work[work["is_frontier"]].sort_values("positive_coverage_mean")
    if len(frontier) > 1:
        ax.plot(
            frontier["positive_coverage_mean"],
            frontier["aps_mean"],
            color=NEUTRAL_COLOR,
            linewidth=1.4,
            linestyle="--",
            label="Pareto frontier",
            zorder=4,
        )

    fallback_offsets = [(10, 10), (10, 24), (10, -20), (-92, 24), (-110, -24)]
    for fallback_offset, (_, row) in zip(fallback_offsets * 3, frontier.iterrows()):
        label_position = _expanded_frontier_label_position(row["combination_id"])
        textcoords = "data" if label_position else "offset points"
        xytext = label_position or fallback_offset
        ax.annotate(
            _combination_label(row["combination_id"]),
            (row["positive_coverage_mean"], row["aps_mean"]),
            xytext=xytext,
            textcoords=textcoords,
            fontsize=10.5,
            color="#22303C",
            arrowprops={
                "arrowstyle": "-",
                "color": NEUTRAL_COLOR,
                "linewidth": 1.0,
                "shrinkA": 3,
                "shrinkB": 4,
            },
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
    ax.set_xlabel("Mean positive coverage", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean APS", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_xlim(0, min(1.02, max(0.15, float(work["positive_coverage_mean"].max()) * 1.18)))
    ax.set_ylim(0, min(1.02, max(0.2, float(work["aps_mean"].max()) * 1.22)))
    _add_figure_heading(
        fig,
        title=title,
        subtitle="Predictors and rank-mean combinations are shown; only Pareto-frontier points are annotated.",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SINGLE_PREDICTOR_COLOR,
            markeredgecolor="black",
            markersize=9,
            label="single predictor",
        )
    ]
    for size in range(min_combination_size, max_combination_size + 1):
        if not bool((work["combination_size"] == size).any()):
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=_combination_size_color(
                    size,
                    min_size=min_combination_size,
                    max_size=max_combination_size,
                ),
                markeredgecolor="black",
                markersize=9,
                label=f"{size}-predictor combination",
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=NEUTRAL_COLOR,
            linestyle="--",
            linewidth=1.9,
            label="Pareto frontier",
        )
    )
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=PLOT_LEGEND_SIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
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
    predictor_top_fraction=DEFAULT_PREDICTOR_TOP_FRACTION,
    logger=None,
):
    out_tables_dir = pathlib.Path(out_tables_dir)
    out_plots_dir = pathlib.Path(out_plots_dir) / "combinations"
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    expanded_summary_df = compute_predictor_combination_summary(
        joined_frames,
        tool_ids=tool_ids,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        max_combination_size=max_combination_size,
        predictor_top_fraction=predictor_top_fraction,
        min_dataset_coverage=0.0,
    )
    table_path = out_tables_dir / "predictor_combination_summary.tsv"
    expanded_summary_df.to_csv(table_path, sep="\t", index=False)
    expanded_plot_path = out_plots_dir / "predictor_combination_expanded_frontier.png"
    write_predictor_combination_expanded_frontier_plot(expanded_summary_df, expanded_plot_path)
    if logger is not None:
        logger(f"Wrote predictor-combination summary: {table_path}")
        logger(f"Wrote expanded predictor-combination frontier: {expanded_plot_path}")
    return {
        "tables": {"predictor_combination_summary": str(table_path)},
        "plots": {
            "predictor_combination_expanded_frontier": str(expanded_plot_path),
        },
    }
