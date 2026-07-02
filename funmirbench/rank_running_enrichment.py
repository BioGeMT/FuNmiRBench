"""GSEA-like running enrichment plots for predictor rank analyses."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funmirbench.evaluate as ev


DEFAULT_GRID_POINTS = 201
DEFAULT_RANK_TYPE = "local"


def _rank_specs_for_frame(frame, *, rank_type):
    rank_specs = ev._rank_distribution_specs(frame, rank_types=(rank_type,))
    return [
        (tool_id, column, column_type)
        for tool_id, column, column_type in rank_specs
        if ev._is_publication_tool(tool_id)
    ]


def _rank_values_for_spec(frame, column, column_type):
    if column_type == "score":
        return ev._rank_scale_scores(frame[column])
    return frame[column].astype(float)


def _running_enrichment_curve(is_positive):
    """Return an unweighted GSEA-like running enrichment curve.

    Hits are GT-positive genes. The curve steps up by 1/N_hits for a hit and
    down by 1/N_background for a background gene. The ranked list is expected
    to be ordered from strongest to weakest prediction before this function is
    called.
    """
    hits = np.asarray(is_positive, dtype=int)
    if hits.size == 0:
        return None
    positive_count = int(hits.sum())
    background_count = int(hits.size - positive_count)
    if positive_count == 0 or background_count == 0:
        return None
    increments = np.where(hits == 1, 1.0 / positive_count, -1.0 / background_count)
    running = np.cumsum(increments)
    x_values = np.arange(1, hits.size + 1, dtype=float) / float(hits.size)
    return x_values, running


def _interpolate_curve(x_values, y_values, grid):
    x_full = np.concatenate([[0.0], np.asarray(x_values, dtype=float)])
    y_full = np.concatenate([[0.0], np.asarray(y_values, dtype=float)])
    return np.interp(grid, x_full, y_full)


def _prepare_ranked_frame(joined, *, fdr_threshold, abs_logfc_threshold, rank_type):
    rank_specs = _rank_specs_for_frame(joined, rank_type=rank_type)
    if not rank_specs:
        return None, []

    keep_cols = ["gene_id", "logFC", "FDR", *[column for _, column, _ in rank_specs]]
    for optional in ("dataset_id", "perturbation", *ev.FDR_AUXILIARY_COLUMNS):
        if optional in joined.columns:
            keep_cols.append(optional)
    keep_cols = list(dict.fromkeys(keep_cols))
    work = ev._filter_usable_gt_rows(joined[keep_cols], fdr_threshold=fdr_threshold)
    if work.empty:
        return None, []

    work = ev._annotate_ground_truth(work)
    work["is_positive"] = ev._positive_mask(
        work,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    ).astype(int)
    return work, rank_specs


def _dataset_id_for_frame(frame, fallback):
    if "dataset_id" in frame.columns and frame["dataset_id"].notna().any():
        return str(frame["dataset_id"].dropna().iloc[0])
    return str(fallback)


def _iter_running_enrichment_rows(
    joined_frames,
    *,
    fdr_threshold,
    abs_logfc_threshold,
    rank_type,
    universe,
    grid,
):
    for dataset_index, joined in enumerate(joined_frames):
        dataset_id = _dataset_id_for_frame(joined, f"dataset_{dataset_index + 1}")
        work, rank_specs = _prepare_ranked_frame(
            joined,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            rank_type=rank_type,
        )
        if work is None or not rank_specs:
            continue

        rank_values_by_tool = {}
        for tool_id, column, column_type in rank_specs:
            rank_values_by_tool[tool_id] = _rank_values_for_spec(work, column, column_type)

        common_mask = None
        if universe == "common_scored":
            common_mask = pd.Series(True, index=work.index)
            for values in rank_values_by_tool.values():
                common_mask &= values.notna()
            if not common_mask.any():
                continue
        elif universe != "all_scored":
            raise ValueError(f"Unsupported enrichment universe: {universe}")

        for tool_id, column, _ in rank_specs:
            values = rank_values_by_tool[tool_id]
            if universe == "common_scored":
                sub = work.loc[common_mask, ["gene_id", "is_positive"]].copy()
                sub["rank_value"] = values.loc[sub.index].astype(float)
            else:
                sub = work.loc[values.notna(), ["gene_id", "is_positive"]].copy()
                sub["rank_value"] = values.loc[sub.index].astype(float)
            if sub.empty:
                continue
            sub = sub.sort_values(["rank_value", "gene_id"], ascending=[False, True], kind="mergesort")
            curve = _running_enrichment_curve(sub["is_positive"].to_numpy(dtype=int))
            if curve is None:
                continue
            x_values, running = curve
            interpolated = _interpolate_curve(x_values, running, grid)
            max_index = int(np.nanargmax(running))
            min_index = int(np.nanargmin(running))
            summary_row = {
                "rank_type": rank_type,
                "universe": universe,
                "dataset_id": dataset_id,
                "tool_id": tool_id,
                "tool_label": ev._tool_label(tool_id),
                "genes_in_universe": int(len(sub)),
                "gt_positives": int(sub["is_positive"].sum()),
                "background_genes": int(len(sub) - sub["is_positive"].sum()),
                "max_enrichment_score": float(running[max_index]),
                "min_enrichment_score": float(running[min_index]),
                "leading_edge_fraction_at_max": float(x_values[max_index]),
                "trailing_edge_fraction_at_min": float(x_values[min_index]),
            }
            for fraction, score in zip(grid, interpolated):
                yield {
                    "rank_type": rank_type,
                    "universe": universe,
                    "dataset_id": dataset_id,
                    "tool_id": tool_id,
                    "tool_label": ev._tool_label(tool_id),
                    "rank_fraction": float(fraction),
                    "running_enrichment_score": float(score),
                }, summary_row


def _running_enrichment_tables(
    joined_frames,
    *,
    fdr_threshold,
    abs_logfc_threshold,
    rank_type,
    universe,
    grid,
):
    curve_rows = []
    summary_by_key = {}
    for curve_row, summary_row in _iter_running_enrichment_rows(
        joined_frames,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        rank_type=rank_type,
        universe=universe,
        grid=grid,
    ):
        curve_rows.append(curve_row)
        key = (summary_row["universe"], summary_row["dataset_id"], summary_row["tool_id"])
        summary_by_key[key] = summary_row
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_by_key.values())


def _aggregate_running_curves(curve_df):
    if curve_df.empty:
        return curve_df
    return (
        curve_df.groupby(["rank_type", "universe", "tool_id", "tool_label", "rank_fraction"], as_index=False)
        .agg(mean_running_enrichment_score=("running_enrichment_score", "mean"))
    )


def _aggregate_running_summary(summary_df):
    if summary_df.empty:
        return summary_df
    return (
        summary_df.groupby(["rank_type", "universe", "tool_id", "tool_label"], as_index=False)
        .agg(
            dataset_count=("dataset_id", "nunique"),
            mean_genes_in_universe=("genes_in_universe", "mean"),
            mean_gt_positives=("gt_positives", "mean"),
            mean_background_genes=("background_genes", "mean"),
            mean_max_enrichment_score=("max_enrichment_score", "mean"),
            median_max_enrichment_score=("max_enrichment_score", "median"),
            mean_min_enrichment_score=("min_enrichment_score", "mean"),
            mean_leading_edge_fraction_at_max=("leading_edge_fraction_at_max", "mean"),
        )
    )


def _plot_running_enrichment(aggregate_curve_df, *, out_path, rank_type, universe):
    if aggregate_curve_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ev._style_axes(ax, grid_axis="both")
    plotted = False
    for tool_id in aggregate_curve_df["tool_id"].drop_duplicates().tolist():
        sub = aggregate_curve_df[aggregate_curve_df["tool_id"] == tool_id].sort_values("rank_fraction")
        if sub.empty:
            continue
        plotted = True
        ax.plot(
            sub["rank_fraction"],
            sub["mean_running_enrichment_score"],
            linewidth=2.5,
            color=ev._tool_color(tool_id),
            label=ev._tool_label(tool_id),
        )
    if not plotted:
        plt.close(fig)
        return False

    if universe == "common_scored":
        universe_label = "restricted to genes scored by all predictors"
        x_label = "Position in ranked list (fraction of common-scored genes)"
    else:
        universe_label = "each predictor on its own scored universe"
        x_label = "Position in ranked list (fraction of scored genes)"

    ax.axhline(0, color="#22303C", linewidth=1.0, alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_xlabel(x_label, fontsize=ev.PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel("Running enrichment score", fontsize=ev.PLOT_AXIS_LABEL_SIZE)
    ev._add_figure_heading(
        fig,
        title=f"{rank_type.capitalize()} rank running enrichment of GT positives",
        subtitle=f"GSEA-like unweighted curve; {universe_label}.",
    )
    ax.legend(frameon=False, fontsize=ev.PLOT_LEGEND_SIZE, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.11, right=0.78)
    ev._save_figure(fig, out_path)
    return True


def write_rank_running_enrichment_outputs(
    joined_frames,
    tables_dir,
    plots_dir,
    *,
    fdr_threshold=0.05,
    abs_logfc_threshold=1.0,
    rank_type=DEFAULT_RANK_TYPE,
    grid_points=DEFAULT_GRID_POINTS,
    logger=None,
):
    """Write GSEA-like rank running enrichment plots and supporting TSVs.

    Two universes are produced:
    - all_scored: each predictor is evaluated on genes it scores.
    - common_scored: all predictors are compared on the shared scored gene universe.
    """
    tables_dir = pathlib.Path(tables_dir)
    plots_dir = pathlib.Path(plots_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, int(grid_points))

    table_paths = {}
    plot_paths = {}
    for universe in ("all_scored", "common_scored"):
        curve_df, summary_df = _running_enrichment_tables(
            joined_frames,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
            rank_type=rank_type,
            universe=universe,
            grid=grid,
        )
        if curve_df.empty or summary_df.empty:
            continue
        aggregate_curve_df = _aggregate_running_curves(curve_df)
        aggregate_summary_df = _aggregate_running_summary(summary_df)

        key = f"{rank_type}_rank_running_enrichment_{universe}"
        curve_path = tables_dir / f"{key}_curves.tsv"
        summary_path = tables_dir / f"{key}_summary.tsv"
        aggregate_path = tables_dir / f"{key}_aggregate.tsv"
        plot_path = plots_dir / f"{key}.png"

        curve_df.to_csv(curve_path, sep="\t", index=False)
        summary_df.to_csv(summary_path, sep="\t", index=False)
        aggregate_summary_df.to_csv(aggregate_path, sep="\t", index=False)
        wrote_plot = _plot_running_enrichment(
            aggregate_curve_df,
            out_path=plot_path,
            rank_type=rank_type,
            universe=universe,
        )
        table_paths[f"{key}_curves"] = str(curve_path)
        table_paths[f"{key}_summary"] = str(summary_path)
        table_paths[f"{key}_aggregate"] = str(aggregate_path)
        if wrote_plot:
            plot_paths[key] = str(plot_path)
            ev._emit_log(logger, f"  Wrote {rank_type} rank running enrichment plot: {plot_path}")
    return table_paths, plot_paths
