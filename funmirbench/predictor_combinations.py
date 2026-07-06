"""Predictor-combination analysis for benchmark reports.

This module tests whether combining predictors improves over individual
predictors. It writes the supporting combination summary table.
"""

from __future__ import annotations

import itertools
import pathlib

import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from funmirbench.evaluate import (
    FDR_AUXILIARY_COLUMNS,
    SCORE_PREFIX,
    _annotate_ground_truth,
    _filter_usable_gt_rows,
    _positive_mask,
    _rank_scale_scores,
    _top_fraction_mask,
)


EXCLUDED_COMBINATION_TOOL_IDS = set()
DEFAULT_MIN_DATASET_COVERAGE = 0.01
DEFAULT_MAX_COMBINATION_SIZE = None
DEFAULT_PREDICTOR_TOP_FRACTION = 0.10
COMBINATION_SUMMARY_COLUMNS = [
    "combination_id",
    "tool_ids",
    "combination_size",
    "contains_control_predictor",
    "ensemble_rule",
    "intersection_rule",
    "dataset_count",
    "coverage_mean",
    "coverage_median",
    "coverage_count",
    "positive_coverage_mean",
    "positive_coverage_median",
    "positive_coverage_count",
    "aps_mean",
    "aps_median",
    "aps_count",
    "pr_auc_mean",
    "pr_auc_median",
    "pr_auc_count",
    "auroc_mean",
    "auroc_median",
    "auroc_count",
    "precision_at_top_100_mean",
    "precision_at_top_100_median",
    "precision_at_top_100_count",
    "intersection_selected_mean",
    "intersection_selected_median",
    "intersection_selected_count",
    "intersection_positives_mean",
    "intersection_positives_median",
    "intersection_positives_count",
    "intersection_background_mean",
    "intersection_background_median",
    "intersection_background_count",
    "intersection_precision_mean",
    "intersection_precision_median",
    "intersection_precision_count",
    "intersection_recovery_mean",
    "intersection_recovery_median",
    "intersection_recovery_count",
    "intersection_coverage_mean",
    "intersection_coverage_median",
    "intersection_coverage_count",
    "delta_aps_vs_best_single",
    "beats_best_single_aps",
]


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
    for optional in ("dataset_id", "mirna", "perturbation", *FDR_AUXILIARY_COLUMNS):
        if optional in joined.columns:
            keep_cols.append(optional)
    work = joined[keep_cols].copy()
    work = _filter_usable_gt_rows(work, fdr_threshold=fdr_threshold)
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
    for optional in ("dataset_id", "mirna", "perturbation", *FDR_AUXILIARY_COLUMNS):
        if optional in joined.columns:
            keep_cols.append(optional)
    work = joined[keep_cols].copy()
    work = _filter_usable_gt_rows(work, fdr_threshold=fdr_threshold)
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
        return pd.DataFrame(columns=COMBINATION_SUMMARY_COLUMNS)
    out = pd.DataFrame(rows)
    single_rows = out[out["combination_size"] == 1]
    single_reference_rows = single_rows[~single_rows["contains_control_predictor"]]
    if single_reference_rows.empty:
        single_reference_rows = single_rows
    best_single_aps = single_reference_rows["aps_mean"].max()
    out["delta_aps_vs_best_single"] = out["aps_mean"] - best_single_aps
    out["beats_best_single_aps"] = out["delta_aps_vs_best_single"] > 0
    out = out.reindex(columns=COMBINATION_SUMMARY_COLUMNS)
    return out.sort_values(
        ["aps_mean", "positive_coverage_mean", "coverage_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


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
    del out_plots_dir
    out_tables_dir.mkdir(parents=True, exist_ok=True)
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
    if logger is not None:
        logger(f"Wrote predictor-combination summary: {table_path}")
    return {
        "tables": {"predictor_combination_summary": str(table_path)},
        "plots": {},
    }
