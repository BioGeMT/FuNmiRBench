"""Build manuscript figures and tables from a FuNmiRBench report directory.

This post-processing script creates the manuscript assets we currently plan to
include: standalone main figures, a combined all-panel figure, one main summary
table, and one supplementary per-dataset metrics table. It does not rerun the
benchmark.

Example
-------
python scripts/build_manuscript_figures_tables.py \
    --report-dir results/20260703_115539 \
    --out-dir manuscript_assets
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

import funmirbench.evaluate as ev

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
MAIN_METRICS = ["coverage", "positive_coverage", "aps", "pr_auc", "auroc", "spearman"]
SUPPLEMENTARY_METRICS = ["coverage", "positive_coverage", "aps", "pr_auc", "auroc", "spearman"]
METRIC_LABELS = {
    "coverage": "Coverage",
    "positive_coverage": "Positive coverage",
    "aps": "APS",
    "pr_auc": "PR-AUC",
    "auroc": "AUROC",
    "spearman": "Spearman",
}
TOOL_COLORS = {
    "targetscan": "#9467BD",
    "mirdb_mirtarget": "#FF7F0E",
    "microt_cnn": "#1F77B4",
    "mirbind2": "#D62728",
    "miraw": "#2CA02C",
}
BACKGROUND_COLOR = "#C9CED6"


def tool_color(tool_id: str) -> str:
    return TOOL_COLORS.get(tool_id, "#7F7F7F")


def style_axes(ax, *, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def save_figure(fig, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def find_table(report_dir: pathlib.Path, filename: str) -> pathlib.Path:
    candidates = [
        report_dir / "tables" / "combined" / filename,
        report_dir / "tables" / filename,
        report_dir / filename,
    ]
    candidates.extend(sorted(report_dir.glob(f"**/{filename}")))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} below {report_dir}")


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def metric_table_long(report_dir: pathlib.Path, metric: str) -> pd.DataFrame:
    table = pd.read_csv(find_table(report_dir, f"{metric}_per_experiment.tsv"), sep="\t")
    id_cols = [col for col in table.columns if col not in TOOL_IDS]
    available_tools = [tool_id for tool_id in TOOL_IDS if tool_id in table.columns]
    long = table.melt(id_vars=id_cols, value_vars=available_tools, var_name="tool_id", value_name=metric)
    long[metric] = pd.to_numeric(long[metric], errors="coerce")
    return long


def load_per_dataset_metrics(report_dir: pathlib.Path) -> pd.DataFrame:
    merged = None
    key_cols = None
    for metric in SUPPLEMENTARY_METRICS:
        try:
            long = metric_table_long(report_dir, metric)
        except FileNotFoundError:
            continue
        id_cols = [col for col in long.columns if col not in {"tool_id", metric}]
        keys = [*id_cols, "tool_id"]
        if merged is None:
            merged = long
            key_cols = keys
        else:
            merged = merged.merge(long[keys + [metric]], on=key_cols, how="outer")
    if merged is None:
        raise FileNotFoundError("No per-experiment metric tables were found.")
    merged.insert(merged.columns.get_loc("tool_id") + 1, "predictor", merged["tool_id"].map(TOOL_LABELS).fillna(merged["tool_id"]))
    return merged


def metric_rows_for_plots(per_dataset: pd.DataFrame) -> pd.DataFrame:
    id_cols = [col for col in per_dataset.columns if col not in set(SUPPLEMENTARY_METRICS)]
    long = per_dataset.melt(id_vars=id_cols, value_vars=[m for m in MAIN_METRICS if m in per_dataset.columns], var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


def write_tables(per_dataset: pd.DataFrame, tables_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    supp = tables_dir / "table_s1_per_dataset_predictor_metrics.tsv"
    per_dataset.to_csv(supp, sep="\t", index=False)

    metric_cols = [m for m in MAIN_METRICS if m in per_dataset.columns]
    summary = per_dataset.groupby(["tool_id", "predictor"])[metric_cols].agg(["mean", "median"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    table1 = tables_dir / "table1_cross_dataset_predictor_summary.tsv"
    summary.to_csv(table1, sep="\t", index=False)
    return {"table1": table1, "table_s1": supp}


def plot_metric_distribution(ax, metric_rows: pd.DataFrame, metric: str):
    sub = metric_rows[metric_rows["metric"] == metric]
    data = [sub.loc[sub["tool_id"] == tool_id, "value"].dropna().to_numpy(float) for tool_id in TOOL_IDS]
    labels = [TOOL_LABELS[tool_id] for tool_id in TOOL_IDS]
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, tool_id in zip(box["boxes"], TOOL_IDS):
        color = tool_color(tool_id)
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.28)
    for median in box["medians"]:
        median.set_color("#22303C")
        median.set_linewidth(1.4)
    for idx, (tool_id, values) in enumerate(zip(TOOL_IDS, data), start=1):
        if values.size:
            jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.size, idx) + jitter,
                values,
                s=18,
                alpha=0.72,
                color=tool_color(tool_id),
                edgecolors="white",
                linewidths=0.3,
            )
    ax.set_title(METRIC_LABELS[metric])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if metric in {"coverage", "positive_coverage", "aps", "pr_auc", "auroc"}:
        ax.set_ylim(0, 1.02)
    else:
        values = np.concatenate([arr for arr in data if arr.size]) if any(arr.size for arr in data) else np.array([0.0])
        lower = min(-0.05, float(np.nanmin(values)) - 0.02)
        upper = max(0.05, float(np.nanmax(values)) + 0.02)
        ax.set_ylim(lower, upper)
        ax.axhline(0.0, color="#606A75", linewidth=0.8, alpha=0.7)
    if metric in {"coverage", "positive_coverage"}:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    style_axes(ax)


def plot_figure1(metric_rows: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    for ax, metric in zip(axes.ravel(), MAIN_METRICS):
        plot_metric_distribution(ax, metric_rows, metric)
    fig.suptitle("Cross-dataset predictor distributions", fontsize=15, y=1.0)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure1_cross_dataset_distributions.png")


def dense_rank_scale(values: pd.Series) -> pd.Series:
    """Normalize ranks to 0-1 within a scored set using average ranks for ties."""
    values = pd.to_numeric(values, errors="coerce")
    ranks = values.rank(method="average", ascending=True)
    count = int(values.notna().sum())
    if count <= 0:
        return pd.Series(np.nan, index=values.index)
    if count == 1:
        return pd.Series(1.0, index=values.index, dtype=float)
    return (ranks - 1.0) / (float(count) - 1.0)


def rank_values(frame: pd.DataFrame, tool_id: str, rank_type: str) -> pd.Series | None:
    col = f"{rank_type}_rank_{tool_id}"
    score = f"score_{tool_id}"
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    if score in frame.columns:
        return dense_rank_scale(frame[score])
    return None


def usable_joined(frame: pd.DataFrame, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    work = ev._filter_usable_gt_rows(frame, fdr_threshold=fdr)
    work = ev._annotate_ground_truth(work)
    work["is_positive"] = ev._positive_mask(
        work,
        fdr_threshold=fdr,
        abs_logfc_threshold=effect_threshold,
    )
    return work


def rank_fraction_table(report_dir: pathlib.Path, *, rank_type: str, bins: int, fdr: float, effect_threshold: float) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    stacked = []
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, rank_type)
            if ranks is not None:
                stacked.append(pd.DataFrame({"tool_id": tool_id, "rank": ranks, "is_positive": frame["is_positive"]}).dropna(subset=["rank"]))
    data = pd.concat(stacked, ignore_index=True)
    rows = []
    for tool_id in TOOL_IDS:
        sub = data[data["tool_id"] == tool_id]
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (sub["rank"] >= left) & ((sub["rank"] <= right) if i == bins - 1 else (sub["rank"] < right))
            in_bin = sub.loc[mask]
            total = int(len(in_bin))
            positives = int(in_bin["is_positive"].sum())
            rows.append({"tool_id": tool_id, "predictor": TOOL_LABELS[tool_id], "rank_bin_mid": (left + right) / 2, "total_count": total, "positive_count": positives, "positive_fraction_within_bin": positives / total if total else np.nan})
    return pd.DataFrame(rows)


def local_rank_distribution_data(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    ranks_by_group: dict[tuple[str, str], list[np.ndarray]] = {}
    for tool_id in TOOL_IDS:
        ranks_by_group[(tool_id, "Background")] = []
        ranks_by_group[(tool_id, "GT positives")] = []

    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, "local")
            if ranks is None:
                continue
            tmp = pd.DataFrame({"rank": ranks, "is_positive": frame["is_positive"].astype(bool)}).dropna(subset=["rank"])
            if tmp.empty:
                continue
            ranks_by_group[(tool_id, "Background")].append(tmp.loc[~tmp["is_positive"], "rank"].to_numpy(float))
            ranks_by_group[(tool_id, "GT positives")].append(tmp.loc[tmp["is_positive"], "rank"].to_numpy(float))

    arrays: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for tool_id in TOOL_IDS:
        for label in ("Background", "GT positives"):
            pieces = ranks_by_group[(tool_id, label)]
            values = np.concatenate(pieces) if pieces else np.array([], dtype=float)
            arrays[(tool_id, label)] = values
            rows.append({
                "tool_id": tool_id,
                "predictor": TOOL_LABELS[tool_id],
                "class": label,
                "n_pairs": int(values.size),
                "median_local_rank": float(np.nanmedian(values)) if values.size else np.nan,
                "mean_local_rank": float(np.nanmean(values)) if values.size else np.nan,
                "q25": float(np.nanquantile(values, 0.25)) if values.size else np.nan,
                "q75": float(np.nanquantile(values, 0.75)) if values.size else np.nan,
            })
    return arrays, pd.DataFrame(rows)


def recovery_table(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float, max_predictions: int = 300) -> pd.DataFrame:
    curves = {tool_id: [] for tool_id in TOOL_IDS}
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        positive_total = int(frame["is_positive"].sum())
        if positive_total <= 0:
            continue
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, "local")
            if ranks is None:
                ranks = rank_values(frame, tool_id, "global")
            if ranks is None:
                continue
            scored = frame.loc[ranks.notna(), ["gene_id", "is_positive"]].copy()
            if scored.empty:
                continue
            scored["rank"] = ranks.loc[scored.index].astype(float)
            scored = scored.sort_values(["rank", "gene_id"], ascending=[False, True], kind="mergesort")
            cumulative = np.cumsum(scored["is_positive"].to_numpy(int)).astype(float) / positive_total
            curve = np.repeat(cumulative[-1], max_predictions)
            observed = min(max_predictions, cumulative.size)
            curve[:observed] = cumulative[:observed]
            curves[tool_id].append(curve)
    rows = []
    for tool_id, tool_curves in curves.items():
        if not tool_curves:
            continue
        mean_curve = np.vstack(tool_curves).mean(axis=0)
        for i, value in enumerate(mean_curve, start=1):
            rows.append({"tool_id": tool_id, "predictor": TOOL_LABELS[tool_id], "prediction_count": i, "recovery_fraction": value})
    return pd.DataFrame(rows)


def plot_fraction(ax, table: pd.DataFrame, *, title: str, xlabel: str):
    for tool_id in TOOL_IDS:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        ax.plot(sub["rank_bin_mid"], sub["positive_fraction_within_bin"] * 100, marker="o", linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GT-positive pairs within bin (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    style_axes(ax, grid_axis="both")


def plot_local_rank_violin(ax, local_rank_arrays: dict[tuple[str, str], np.ndarray], *, seed: int = 42):
    rng = np.random.default_rng(seed)
    positions = []
    violin_data = []
    facecolors = []
    for i, tool_id in enumerate(TOOL_IDS, start=1):
        positions.extend([i - 0.18, i + 0.18])
        for label in ("Background", "GT positives"):
            values = local_rank_arrays[(tool_id, label)]
            max_points = 4000 if label == "Background" else 2000
            if values.size > max_points:
                values = rng.choice(values, size=max_points, replace=False)
            violin_data.append(values)
            facecolors.append(BACKGROUND_COLOR if label == "Background" else tool_color(tool_id))
    parts = ax.violinplot(violin_data, positions=positions, widths=0.28, showmeans=False, showmedians=False, showextrema=False, points=100)
    for body, color in zip(parts["bodies"], facecolors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.60)

    for i, tool_id in enumerate(TOOL_IDS, start=1):
        for pos, label in ((i - 0.18, "Background"), (i + 0.18, "GT positives")):
            values = local_rank_arrays[(tool_id, label)]
            if values.size == 0:
                continue
            q1 = np.nanquantile(values, 0.25)
            median = np.nanmedian(values)
            q3 = np.nanquantile(values, 0.75)
            ax.vlines(pos, q1, q3, color="black", linewidth=1.8, zorder=3)
            ax.hlines(median, pos - 0.095, pos + 0.095, color="black", linewidth=2.0, zorder=4)
    ax.set_xlim(0.4, len(TOOL_IDS) + 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(1, len(TOOL_IDS) + 1))
    ax.set_xticklabels([TOOL_LABELS[tool_id] for tool_id in TOOL_IDS], rotation=18, ha="right")
    ax.set_title("Local-rank distributions")
    ax.set_ylabel("Local normalized rank\n(0 = weakest, 1 = strongest)")
    style_axes(ax, grid_axis="y")


def plot_recovery(ax, recovery: pd.DataFrame):
    for tool_id in TOOL_IDS:
        sub = recovery[recovery["tool_id"] == tool_id]
        ax.plot(sub["prediction_count"], sub["recovery_fraction"] * 100, linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
    ax.set_title("GT-positive recovery")
    ax.set_xlabel("Predicted targets per dataset")
    ax.set_ylabel("Mean GT-positive recovery (%)")
    ax.set_ylim(bottom=0)
    style_axes(ax, grid_axis="both")


def plot_figure2(local: pd.DataFrame, local_rank_arrays: dict[tuple[str, str], np.ndarray], recovery: pd.DataFrame, figures_dir: pathlib.Path):
    fig = plt.figure(figsize=(15.5, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.15, 1.0], hspace=0.43, wspace=0.30)
    ax_local = fig.add_subplot(grid[0, 0])
    ax_violin = fig.add_subplot(grid[0, 1])
    ax_recovery = fig.add_subplot(grid[1, 0])
    ax_legend = fig.add_subplot(grid[1, 1])

    plot_fraction(ax_local, local, title="Local rank enrichment", xlabel="Local normalized rank bin")
    plot_local_rank_violin(ax_violin, local_rank_arrays)
    plot_recovery(ax_recovery, recovery)

    for ax, label in ((ax_local, "A"), (ax_violin, "B"), (ax_recovery, "C")):
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    ax_legend.axis("off")
    predictor_handles = [
        Line2D([0], [0], color=tool_color(tool_id), lw=2.5, marker="o", label=TOOL_LABELS[tool_id])
        for tool_id in TOOL_IDS
    ]
    class_handles = [
        Patch(facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_COLOR, alpha=0.60, label="Background pairs"),
        Patch(facecolor="#777777", edgecolor="#777777", alpha=0.60, label="GT-positive pairs in panel B"),
    ]
    legend1 = ax_legend.legend(handles=predictor_handles, title="Predictor", frameon=False, loc="upper left", fontsize=10, title_fontsize=11)
    ax_legend.add_artist(legend1)
    ax_legend.legend(handles=class_handles, title="Violin classes", frameon=False, loc="lower left", fontsize=10, title_fontsize=11)

    fig.suptitle("Functional target enrichment and recovery across local predictor ranks", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return save_figure(fig, figures_dir / "figure2_rank_enrichment_recovery.png")


def centered_table(report_dir: pathlib.Path, *, anchor_tool: str, fdr: float, effect_threshold: float, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    stacked = []
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        anchor = rank_values(frame, anchor_tool, "local")
        if anchor is None:
            continue
        frame = frame.loc[anchor.notna()].copy()
        anchor = anchor.loc[frame.index].astype(float)
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, "local")
            scored = ranks.notna() if ranks is not None else pd.Series(False, index=frame.index)
            stacked.append(pd.DataFrame({"tool_id": tool_id, "anchor_rank": anchor, "is_positive": frame["is_positive"], "scored": scored.astype(bool)}))
    data = pd.concat(stacked, ignore_index=True)
    rows = []
    for tool_id in TOOL_IDS:
        sub = data[data["tool_id"] == tool_id]
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (sub["anchor_rank"] >= left) & ((sub["anchor_rank"] <= right) if i == bins - 1 else (sub["anchor_rank"] < right))
            in_bin = sub.loc[mask]
            scored = in_bin[in_bin["scored"]]
            total = int(len(in_bin))
            scored_count = int(len(scored))
            positives = int(scored["is_positive"].sum())
            rows.append({"tool_id": tool_id, "predictor": TOOL_LABELS[tool_id], "rank_bin_mid": (left + right) / 2, "anchor_bin_total": total, "predictor_scored_count": scored_count, "coverage_fraction": scored_count / total if total else np.nan, "positive_fraction_within_scored": positives / scored_count if scored_count else np.nan})
    return pd.DataFrame(rows)


def plot_figure3(table: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 9.6), sharex=True)
    for tool_id in TOOL_IDS:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        axes[0].plot(sub["rank_bin_mid"], sub["positive_fraction_within_scored"] * 100, marker="o", linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
        axes[1].plot(sub["rank_bin_mid"], sub["coverage_fraction"] * 100, marker="o", linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
    axes[0].set_title("TargetScan-centered GT-positive enrichment")
    axes[0].set_ylabel("GT positives among scored genes (%)")
    axes[0].set_ylim(bottom=0)
    style_axes(axes[0], grid_axis="both")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[1].set_title("Predictor coverage across TargetScan bins")
    axes[1].set_xlabel("TargetScan local rank bin (0 = weakest, 1 = strongest)")
    axes[1].set_ylabel("Genes in bin scored by predictor (%)")
    axes[1].set_ylim(0, 105)
    style_axes(axes[1], grid_axis="both")
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure3_targetscan_centered.png")


def plot_figure_all(
    metric_rows: pd.DataFrame,
    local: pd.DataFrame,
    local_rank_arrays: dict[tuple[str, str], np.ndarray],
    recovery: pd.DataFrame,
    targetscan: pd.DataFrame,
    figures_dir: pathlib.Path,
):
    fig = plt.figure(figsize=(18, 20.0))
    grid = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.0, 1.05, 1.0], hspace=0.52, wspace=0.40)

    panel_labels = "ABCDEFGHIJK"
    for index, metric in enumerate(MAIN_METRICS):
        row = index // 3
        start_col = (index % 3) * 2
        ax = fig.add_subplot(grid[row, start_col:start_col + 2])
        plot_metric_distribution(ax, metric_rows, metric)
        ax.text(-0.16, 1.10, panel_labels[index], transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    ax_local = fig.add_subplot(grid[2, 0:2])
    plot_fraction(ax_local, local, title="Local rank enrichment", xlabel="Local normalized rank bin")
    ax_local.text(-0.16, 1.10, "G", transform=ax_local.transAxes, fontsize=14, fontweight="bold", va="top")

    ax_violin = fig.add_subplot(grid[2, 2:4])
    plot_local_rank_violin(ax_violin, local_rank_arrays)
    ax_violin.text(-0.16, 1.10, "H", transform=ax_violin.transAxes, fontsize=14, fontweight="bold", va="top")

    ax_recovery = fig.add_subplot(grid[2, 4:6])
    plot_recovery(ax_recovery, recovery)
    ax_recovery.legend(frameon=False, fontsize=9)
    ax_recovery.text(-0.16, 1.10, "I", transform=ax_recovery.transAxes, fontsize=14, fontweight="bold", va="top")

    ax_targetscan_enrichment = fig.add_subplot(grid[3, 0:3])
    ax_targetscan_coverage = fig.add_subplot(grid[3, 3:6])
    for tool_id in TOOL_IDS:
        sub = targetscan[targetscan["tool_id"] == tool_id].sort_values("rank_bin_mid")
        ax_targetscan_enrichment.plot(
            sub["rank_bin_mid"],
            sub["positive_fraction_within_scored"] * 100,
            marker="o",
            linewidth=2,
            color=tool_color(tool_id),
            label=TOOL_LABELS[tool_id],
        )
        ax_targetscan_coverage.plot(
            sub["rank_bin_mid"],
            sub["coverage_fraction"] * 100,
            marker="o",
            linewidth=2,
            color=tool_color(tool_id),
            label=TOOL_LABELS[tool_id],
        )
    ax_targetscan_enrichment.set_title("TargetScan-centered GT-positive enrichment")
    ax_targetscan_enrichment.set_xlabel("TargetScan local rank bin (0 = weakest, 1 = strongest)")
    ax_targetscan_enrichment.set_ylabel("GT positives among scored genes (%)")
    ax_targetscan_enrichment.set_ylim(bottom=0)
    style_axes(ax_targetscan_enrichment, grid_axis="both")
    ax_targetscan_enrichment.text(-0.08, 1.10, "J", transform=ax_targetscan_enrichment.transAxes, fontsize=14, fontweight="bold", va="top")

    ax_targetscan_coverage.set_title("Predictor coverage across TargetScan bins")
    ax_targetscan_coverage.set_xlabel("TargetScan local rank bin (0 = weakest, 1 = strongest)")
    ax_targetscan_coverage.set_ylabel("Genes in bin scored by predictor (%)")
    ax_targetscan_coverage.set_ylim(0, 105)
    style_axes(ax_targetscan_coverage, grid_axis="both")
    ax_targetscan_coverage.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax_targetscan_coverage.text(-0.08, 1.10, "K", transform=ax_targetscan_coverage.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.suptitle("FuNmiRBench manuscript figure panels", fontsize=17, y=0.995)
    return save_figure(fig, figures_dir / "figure_all_manuscript_panels.png")


def build_assets(report_dir: pathlib.Path, out_dir: pathlib.Path, *, fdr: float, effect_threshold: float):
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    per_dataset = load_per_dataset_metrics(report_dir)
    metrics_long = metric_rows_for_plots(per_dataset)
    outputs.update(write_tables(per_dataset, tables_dir))
    outputs["figure1"] = plot_figure1(metrics_long, figures_dir)

    local = rank_fraction_table(report_dir, rank_type="local", bins=10, fdr=fdr, effect_threshold=effect_threshold)
    local_rank_arrays, local_rank_summary = local_rank_distribution_data(report_dir, fdr=fdr, effect_threshold=effect_threshold)
    local_rank_summary_path = tables_dir / "table_s2_local_rank_background_positive_summary.tsv"
    local_rank_summary.to_csv(local_rank_summary_path, sep="\t", index=False)
    outputs["table_s2"] = local_rank_summary_path
    recovery = recovery_table(report_dir, fdr=fdr, effect_threshold=effect_threshold)
    outputs["figure2"] = plot_figure2(local, local_rank_arrays, recovery, figures_dir)

    targetscan = centered_table(report_dir, anchor_tool="targetscan", fdr=fdr, effect_threshold=effect_threshold)
    outputs["figure3"] = plot_figure3(targetscan, figures_dir)
    outputs["figure_all"] = plot_figure_all(metrics_long, local, local_rank_arrays, recovery, targetscan, figures_dir)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Create manuscript figures and tables from a FuNmiRBench report.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    args = parser.parse_args()
    outputs = build_assets(args.report_dir, args.out_dir, fdr=args.fdr_threshold, effect_threshold=args.effect_threshold)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
