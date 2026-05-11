"""Cross-dataset metric tables and summary plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import funmirbench.evaluate as ev


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
        ev._emit_log(logger, f"  Wrote {metric_name} table: {out_path}")
    return out_paths


def _plot_cross_dataset_metric_distribution(metrics_df, *, metric_name, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tool_ids = list(metrics_df["tool_id"].drop_duplicates())
    positions = np.arange(len(tool_ids), dtype=float)
    ev._style_axes(ax, grid_axis="y")
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
    for patch, tool_id in zip(box["boxes"], tool_ids):
        color = ev._tool_color(tool_id)
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
            color=ev._tool_color(tool_ids[tool_index]),
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    ymin, ymax = ev._metric_plot_limits(metric_name)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(metric_name.upper(), fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([ev._tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")
    ax.set_title(
        f"Cross-dataset {metric_name.upper()} distribution",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    ev._save_figure(fig, out_path)


def _rank_distribution_metadata(rank_type):
    if rank_type == "local":
        return {
            "title": "Positive vs background local rank distributions",
            "axis_label": "Local rank within dataset",
            "subtitle": (
                "Aggregated across datasets; dense rank is recomputed within each dataset. "
                "GT positives use each predictor's color, background genes are gray."
            ),
        }
    if rank_type == "global":
        return {
            "title": "Positive vs background global rank distributions",
            "axis_label": "Global rank across predictor file",
            "subtitle": (
                "Aggregated across datasets; dense rank comes from each predictor's full standardized file. "
                "GT positives use each predictor's color, background genes are gray."
            ),
        }
    raise ValueError(f"Unsupported rank distribution type: {rank_type}")


def _plot_rank_class_distributions(
    joined_frames, *, out_path, fdr_threshold, abs_logfc_threshold, rank_type
):
    combined = pd.concat(joined_frames, ignore_index=True)
    rank_specs = ev._rank_distribution_specs(combined, rank_types=(rank_type,))
    if not rank_specs:
        return False

    keep_cols = ["logFC", "FDR", *[column for _, column, _ in rank_specs]]
    for optional in ("dataset_id", "perturbation"):
        if optional in combined.columns:
            keep_cols.append(optional)
    work = combined[keep_cols].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    if work.empty:
        return False
    work = ev._annotate_ground_truth(work)
    work["is_positive"] = ev._positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)

    tool_ids = []
    positive_data = []
    background_data = []
    for tool_id, column, column_type in rank_specs:
        if column_type == "score":
            values = ev._rank_scale_scores(work[column])
        else:
            values = work[column].astype(float)
        pos_values = values.loc[work["is_positive"] == 1].dropna().astype(float).tolist()
        bg_values = values.loc[work["is_positive"] == 0].dropna().astype(float).tolist()
        tool_ids.append(tool_id)
        positive_data.append(pos_values)
        background_data.append(bg_values)
    if not any(positive_data) or not any(background_data):
        return False

    fig, ax = plt.subplots(figsize=(max(8.4, len(tool_ids) * 1.7), 5.8))
    ev._style_axes(ax, grid_axis="y")
    positions = np.arange(len(tool_ids), dtype=float) * 1.35
    negative_positions = positions - 0.18
    positive_positions = positions + 0.18
    colors = [ev._tool_color(tool_id) for tool_id in tool_ids]
    valid_bg_positions = [pos for pos, values in zip(negative_positions, background_data) if values]
    valid_bg_data = [values for values in background_data if values]
    valid_pos_positions = [pos for pos, values in zip(positive_positions, positive_data) if values]
    valid_pos_data = [values for values in positive_data if values]

    if valid_bg_data:
        bg_violin = ax.violinplot(
            valid_bg_data,
            positions=valid_bg_positions,
            widths=0.30,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body in bg_violin["bodies"]:
            body.set_facecolor(ev.NEGATIVE_COLOR)
            body.set_edgecolor(ev.NEGATIVE_COLOR)
            body.set_alpha(0.45)
        bg_violin["cmedians"].set_color("#22303C")
        bg_violin["cmedians"].set_linewidth(1.5)

    if valid_pos_data:
        pos_violin = ax.violinplot(
            valid_pos_data,
            positions=valid_pos_positions,
            widths=0.30,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body, color in zip(pos_violin["bodies"], [c for c, values in zip(colors, positive_data) if values]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.40)
        pos_violin["cmedians"].set_color("#22303C")
        pos_violin["cmedians"].set_linewidth(1.5)

    meta = _rank_distribution_metadata(rank_type)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(meta["axis_label"], fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([ev._tool_label(tool_id) for tool_id in tool_ids], rotation=45, ha="right")
    ax.set_title(
        meta["title"],
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(
        0.125,
        0.955,
        meta["subtitle"],
        fontsize=9,
        color=ev.NEUTRAL_COLOR,
    )
    ax.legend(
        handles=[
            Patch(facecolor=ev.NEGATIVE_COLOR, edgecolor=ev.NEGATIVE_COLOR, alpha=0.45, label="Background genes"),
            Patch(
                facecolor="#C8D6EA",
                edgecolor="#6E89A8",
                alpha=0.50,
                label="GT positives (predictor color)",
            ),
        ],
        frameon=False,
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ev._save_figure(fig, out_path)
    return True


def _plot_positive_recovery_by_prediction_count(
    joined_frames,
    *,
    out_path,
    fdr_threshold,
    abs_logfc_threshold,
    max_predictions=300,
):
    combined = pd.concat(joined_frames, ignore_index=True)
    rank_specs = ev._rank_distribution_specs(combined, rank_types=("local", "global", "score"))
    if not rank_specs:
        return False

    dataset_col = "dataset_id" if "dataset_id" in combined.columns else None
    keep_cols = ["gene_id", "logFC", "FDR", *[column for _, column, _ in rank_specs]]
    for optional in ("dataset_id", "perturbation"):
        if optional in combined.columns:
            keep_cols.append(optional)
    keep_cols = list(dict.fromkeys(keep_cols))
    work = combined[keep_cols].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    if work.empty:
        return False

    work = ev._annotate_ground_truth(work)
    work["is_positive"] = ev._positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
    groups = (
        list(work.groupby(dataset_col, sort=True))
        if dataset_col
        else [("all_datasets", work)]
    )

    max_predictions = int(max_predictions)
    if max_predictions <= 0:
        return False
    x_values = np.arange(1, max_predictions + 1, dtype=int)
    plotted = False

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ev._style_axes(ax, grid_axis="both")
    for tool_id, column, column_type in rank_specs:
        dataset_curves = []
        for _, dataset_frame in groups:
            if column_type == "score":
                values = ev._rank_scale_scores(dataset_frame[column])
            else:
                values = dataset_frame[column].astype(float)
            scored = dataset_frame.loc[values.notna(), ["gene_id", "is_positive"]].copy()
            if scored.empty:
                continue
            scored["rank_value"] = values.loc[scored.index].astype(float)
            scored = scored.sort_values(
                ["rank_value", "gene_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            hits = scored["is_positive"].to_numpy(dtype=int)
            if hits.size == 0:
                continue
            cumulative = np.cumsum(hits)
            curve = np.repeat(float(cumulative[-1]), max_predictions)
            observed = min(max_predictions, cumulative.size)
            curve[:observed] = cumulative[:observed].astype(float)
            dataset_curves.append(curve)

        if not dataset_curves:
            continue
        mean_curve = np.vstack(dataset_curves).mean(axis=0)
        if not np.isfinite(mean_curve).any():
            continue
        plotted = True
        ax.plot(
            x_values,
            mean_curve,
            linewidth=2.0,
            color=ev._tool_color(tool_id),
            label=ev._tool_label(tool_id),
        )

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlim(1, max_predictions)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted targets per dataset", fontsize=10)
    ax.set_ylabel("Mean GT positives recovered", fontsize=10)
    ax.set_title(
        "GT-positive recovery by prediction count",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    fig.text(
        0.125,
        0.955,
        (
            "Figure 4-style cumulative recovery: each curve counts GT positives recovered "
            "as more top-ranked predictions per dataset are admitted."
        ),
        fontsize=9,
        color=ev.NEUTRAL_COLOR,
    )
    ax.legend(frameon=False, fontsize=8.6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ev._save_figure(fig, out_path)
    return True


def write_cross_dataset_summaries(
    metric_rows,
    tables_dir,
    plots_dir,
    *,
    joined_frames=None,
    fdr_threshold=0.05,
    abs_logfc_threshold=1.0,
    tool_labels=None,
    logger=None,
):
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ev._set_tool_labels(tool_labels)
    metric_plots_dir = plots_dir / "metrics"
    rank_plots_dir = plots_dir / "ranks"
    for path in (metric_plots_dir, rank_plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        return {
            "tables": {},
            "plots": {},
        }
    ev._set_tool_colors(metrics_df["tool_id"].drop_duplicates().tolist())

    metric_names = ["coverage", "positive_coverage", "aps", "pr_auc", "spearman", "auroc"]
    summary = metrics_df.groupby("tool_id")[metric_names].agg(["count", "mean", "median", "std", "min", "max"])
    summary.columns = [f"{metric_name}_{stat_name}" for metric_name, stat_name in summary.columns]
    summary = summary.reset_index()
    summary_path = tables_dir / "cross_dataset_predictor_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    ev._emit_log(logger, f"  Wrote cross-dataset summary table: {summary_path}")

    distribution_paths = {}
    for metric_name in metric_names:
        metric_path = metric_plots_dir / f"cross_dataset_{metric_name}_distribution.png"
        _plot_cross_dataset_metric_distribution(
            metrics_df,
            metric_name=metric_name,
            out_path=metric_path,
        )
        distribution_paths[f"cross_dataset_{metric_name}_distribution"] = str(metric_path)
        ev._emit_log(logger, f"  Wrote cross-dataset {metric_name} distribution: {metric_path}")

    rank_distribution_paths = {}
    if joined_frames:
        for rank_type, plot_key, filename in (
            (
                "local",
                "positive_background_local_rank_distributions",
                "positive_background_local_rank_distributions.png",
            ),
            (
                "global",
                "positive_background_global_rank_distributions",
                "positive_background_global_rank_distributions.png",
            ),
        ):
            rank_distribution_path = rank_plots_dir / filename
            wrote_rank_distribution = _plot_rank_class_distributions(
                joined_frames,
                out_path=rank_distribution_path,
                fdr_threshold=fdr_threshold,
                abs_logfc_threshold=abs_logfc_threshold,
                rank_type=rank_type,
            )
            if wrote_rank_distribution:
                rank_distribution_paths[plot_key] = str(rank_distribution_path)
                ev._emit_log(logger, f"  Wrote {rank_type} rank distribution plot: {rank_distribution_path}")
        recovery_path = rank_plots_dir / "positive_recovery_by_prediction_count.png"
        wrote_recovery = _plot_positive_recovery_by_prediction_count(
            joined_frames,
            out_path=recovery_path,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
        )
        if wrote_recovery:
            rank_distribution_paths["positive_recovery_by_prediction_count"] = str(recovery_path)
            ev._emit_log(logger, f"  Wrote positive recovery plot: {recovery_path}")

    return {
        "tables": {
            "cross_dataset_predictor_summary": str(summary_path),
        },
        "plots": {
            **distribution_paths,
            **rank_distribution_paths,
        },
    }
