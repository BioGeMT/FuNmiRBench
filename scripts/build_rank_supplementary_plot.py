"""Build the supplementary local-rank enrichment/recovery figure.

This script creates a three-panel supplementary diagnostic figure:

A. Local-rank enrichment: fraction of GT-positive pairs within local-rank bins.
B. Local-rank distributions: background versus GT-positive local ranks.
C. GT-positive recovery: mean fraction of GT positives recovered as top-ranked
   predictions are admitted per dataset.

Local ranks are recomputed inside each dataset and predictor from raw score
columns over scored pairs only, using average ranks for tied scores. The plot is
intentionally focused on local predictor ranks and does not rerun the full
benchmark. It consumes an existing FuNmiRBench report directory containing
per-dataset joined.tsv files.
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

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
TOOL_COLORS = {
    "targetscan": "#9467BD",
    "mirdb_mirtarget": "#FF7F0E",
    "microt_cnn": "#1F77B4",
    "mirbind2": "#D62728",
    "miraw": "#2CA02C",
}
BACKGROUND_COLOR = "#CBD2DB"
BACKGROUND_EDGE = "#A8B0BA"
TEXT_COLOR = "#1F2933"
GRID_COLOR = "#D8DEE9"


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def dataset_id_from_joined(frame: pd.DataFrame, path: pathlib.Path) -> str:
    if "dataset_id" in frame.columns and not frame.empty:
        return str(frame["dataset_id"].iloc[0])
    return path.parent.name


def expected_effect(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    perturbation = frame.get("perturbation", pd.Series("", index=frame.index)).astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    oe = perturbation.isin(["OE", "OVEREXPRESSION"])
    kdko = perturbation.isin(["KO", "KD", "KNOCKOUT", "KNOCKDOWN"])
    effect.loc[oe] = -logfc.loc[oe]
    effect.loc[kdko] = logfc.loc[kdko]
    return effect


def usable_frame(frame: pd.DataFrame, *, fdr_threshold: float, effect_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
    frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
    keep = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
    frame = frame.loc[keep].copy()
    frame["expected_effect"] = expected_effect(frame)
    frame["is_positive"] = ((frame["FDR_num"] < fdr_threshold) & (frame["expected_effect"] > effect_threshold)).astype(int)
    return frame


def local_rank_values(frame: pd.DataFrame, tool_id: str) -> pd.Series | None:
    """Recompute local normalized ranks from raw scores within one dataset.

    Only scored miRNA-gene pairs for the predictor are ranked. Tied scores receive
    average ranks. The returned rank is normalized to [0, 1], with 0 as the
    weakest scored pair and 1 as the strongest scored pair.
    """
    score_col = f"score_{tool_id}"
    if score_col not in frame.columns:
        return None

    scores = pd.to_numeric(frame[score_col], errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = scores.notna()
    n_scored = int(valid.sum())
    if n_scored == 0:
        return out
    if n_scored == 1:
        out.loc[valid] = 0.5
        return out

    average_rank = scores.loc[valid].rank(method="average", ascending=True)
    out.loc[valid] = (average_rank - 1.0) / float(n_scored - 1)
    return out


def load_rank_table(report_dir: pathlib.Path, *, fdr_threshold: float, effect_threshold: float) -> pd.DataFrame:
    rows = []
    for path in joined_paths(report_dir):
        raw = pd.read_csv(path, sep="\t")
        dataset_id = dataset_id_from_joined(raw, path)
        frame = usable_frame(raw, fdr_threshold=fdr_threshold, effect_threshold=effect_threshold)
        if frame.empty:
            continue
        genes = frame["gene_id"].astype(str) if "gene_id" in frame.columns else frame.index.astype(str)
        for tool_id in TOOL_IDS:
            values = local_rank_values(frame, tool_id)
            if values is None:
                continue
            sub = pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "gene_id": genes,
                    "tool_id": tool_id,
                    "tool_label": TOOL_LABELS[tool_id],
                    "rank_value": pd.to_numeric(values, errors="coerce"),
                    "is_positive": frame["is_positive"].astype(int),
                }
            ).dropna(subset=["rank_value"])
            if sub.empty:
                continue
            sub["rank_value"] = sub["rank_value"].clip(0.0, 1.0)
            rows.append(sub)
    if not rows:
        raise ValueError(f"No local-rank rows were created below {report_dir}")
    return pd.concat(rows, ignore_index=True)


def enrichment_by_bin(rank_df: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    for tool_id in TOOL_IDS:
        sub = rank_df[rank_df["tool_id"] == tool_id]
        if sub.empty:
            continue
        for left, right in zip(bins[:-1], bins[1:]):
            if right == 1.0:
                in_bin = (sub["rank_value"] >= left) & (sub["rank_value"] <= right)
            else:
                in_bin = (sub["rank_value"] >= left) & (sub["rank_value"] < right)
            bin_frame = sub.loc[in_bin]
            total = int(len(bin_frame))
            positive = int(bin_frame["is_positive"].sum())
            rows.append(
                {
                    "tool_id": tool_id,
                    "tool_label": TOOL_LABELS[tool_id],
                    "rank_bin_start": float(left),
                    "rank_bin_end": float(right),
                    "rank_bin_mid": float((left + right) / 2.0),
                    "total_count": total,
                    "positive_count": positive,
                    "positive_fraction_within_bin": positive / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def recovery_curves(rank_df: pd.DataFrame, *, max_predictions: int) -> pd.DataFrame:
    x_values = np.arange(1, int(max_predictions) + 1, dtype=int)
    rows = []
    for tool_id in TOOL_IDS:
        dataset_curves = []
        for _, dataset_frame in rank_df[rank_df["tool_id"] == tool_id].groupby("dataset_id", sort=True):
            positive_total = int(dataset_frame["is_positive"].sum())
            if positive_total <= 0:
                continue
            ordered = dataset_frame.sort_values(["rank_value", "gene_id"], ascending=[False, True], kind="mergesort")
            hits = ordered["is_positive"].to_numpy(dtype=int)
            if hits.size == 0:
                continue
            cumulative = np.cumsum(hits).astype(float) / float(positive_total)
            curve = np.repeat(float(cumulative[-1]), int(max_predictions))
            observed = min(int(max_predictions), cumulative.size)
            curve[:observed] = cumulative[:observed]
            dataset_curves.append(curve)
        if not dataset_curves:
            continue
        mean_curve = np.vstack(dataset_curves).mean(axis=0)
        rows.append(pd.DataFrame({"tool_id": tool_id, "tool_label": TOOL_LABELS[tool_id], "prediction_count": x_values, "mean_recovery_fraction": mean_curve}))
    if not rows:
        return pd.DataFrame(columns=["tool_id", "tool_label", "prediction_count", "mean_recovery_fraction"])
    return pd.concat(rows, ignore_index=True)


def style_axes(ax, *, grid_axis="both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11, colors=TEXT_COLOR)


def add_panel_label(fig, ax, label: str, *, x_offset: float = -0.055, y_offset: float = 0.03) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 + x_offset, bbox.y1 + y_offset, label, fontsize=18, fontweight="bold", ha="left", va="top", color="black")


def plot_panel_a(ax, enrichment_df: pd.DataFrame) -> None:
    style_axes(ax, grid_axis="both")
    for tool_id in TOOL_IDS:
        sub = enrichment_df[enrichment_df["tool_id"] == tool_id]
        if sub.empty:
            continue
        ax.plot(
            sub["rank_bin_mid"],
            sub["positive_fraction_within_bin"],
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            color=TOOL_COLORS[tool_id],
            label=TOOL_LABELS[tool_id],
        )
    ax.set_title("Local rank enrichment", fontsize=14)
    ax.set_xlabel("Local normalized rank bin", fontsize=12)
    ax.set_ylabel("GT-positive pairs within bin (%)", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(0.05, float(enrichment_df["positive_fraction_within_bin"].max(skipna=True) or 0.0) * 1.08))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def plot_panel_b(ax, rank_df: pd.DataFrame) -> None:
    style_axes(ax, grid_axis="y")
    positions = np.arange(len(TOOL_IDS), dtype=float)
    background_positions = positions - 0.16
    positive_positions = positions + 0.16
    background_data = []
    positive_data = []
    for tool_id in TOOL_IDS:
        sub = rank_df[rank_df["tool_id"] == tool_id]
        background_data.append(sub.loc[sub["is_positive"] == 0, "rank_value"].dropna().to_numpy(float))
        positive_data.append(sub.loc[sub["is_positive"] == 1, "rank_value"].dropna().to_numpy(float))

    bg_violin = ax.violinplot(background_data, positions=background_positions, widths=0.24, showmeans=False, showmedians=False, showextrema=False)
    for body in bg_violin["bodies"]:
        body.set_facecolor(BACKGROUND_COLOR)
        body.set_edgecolor(BACKGROUND_EDGE)
        body.set_alpha(0.55)

    pos_violin = ax.violinplot(positive_data, positions=positive_positions, widths=0.24, showmeans=False, showmedians=False, showextrema=False)
    for body, tool_id in zip(pos_violin["bodies"], TOOL_IDS):
        body.set_facecolor(TOOL_COLORS[tool_id])
        body.set_edgecolor(TOOL_COLORS[tool_id])
        body.set_alpha(0.55)

    for pos, values in zip(background_positions, background_data):
        if values.size:
            q25, med, q75 = np.nanpercentile(values, [25, 50, 75])
            ax.vlines(pos, q25, q75, color="black", linewidth=2.0, zorder=4)
            ax.plot(pos, med, marker="o", color="black", markersize=4.2, zorder=5)
    for pos, values in zip(positive_positions, positive_data):
        if values.size:
            q25, med, q75 = np.nanpercentile(values, [25, 50, 75])
            ax.vlines(pos, q25, q75, color="black", linewidth=2.0, zorder=4)
            ax.plot(pos, med, marker="o", color="black", markersize=4.2, zorder=5)

    ax.set_title("Local-rank distributions", fontsize=14)
    ax.set_ylabel("Local normalized rank\n(0 = weakest, 1 = strongest)", fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels([TOOL_LABELS[tool_id] for tool_id in TOOL_IDS], rotation=18, ha="right", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.6, len(TOOL_IDS) - 0.4)


def plot_panel_c(ax, recovery_df: pd.DataFrame, *, max_predictions: int) -> None:
    style_axes(ax, grid_axis="both")
    for tool_id in TOOL_IDS:
        sub = recovery_df[recovery_df["tool_id"] == tool_id]
        if sub.empty:
            continue
        ax.plot(
            sub["prediction_count"],
            sub["mean_recovery_fraction"],
            linewidth=2.4,
            color=TOOL_COLORS[tool_id],
            label=TOOL_LABELS[tool_id],
        )
    ax.set_title("GT-positive recovery", fontsize=14)
    ax.set_xlabel("Predicted targets per dataset", fontsize=12)
    ax.set_ylabel("Mean GT-positive recovery (%)", fontsize=12)
    ax.set_xlim(1, int(max_predictions))
    ymax = max(0.01, float(recovery_df["mean_recovery_fraction"].max(skipna=True) or 0.0) * 1.08)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def plot_legend_panel(ax) -> None:
    ax.axis("off")
    predictor_handles = [
        Line2D([0], [0], color=TOOL_COLORS[tool_id], marker="o", linewidth=2.4, markersize=6, label=TOOL_LABELS[tool_id])
        for tool_id in TOOL_IDS
    ]
    class_handles = [
        Patch(facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_EDGE, alpha=0.55, label="Background pairs"),
        Patch(facecolor="#999999", edgecolor="#666666", alpha=0.55, label="GT-positive pairs in panel B"),
        Line2D([0], [0], color="black", marker="o", linewidth=2.0, markersize=4.5, label="Median + IQR"),
    ]
    first = ax.legend(handles=predictor_handles, title="Predictor", frameon=False, loc="upper left", bbox_to_anchor=(0.00, 0.95), fontsize=11, title_fontsize=12)
    ax.add_artist(first)
    ax.legend(handles=class_handles, title="Violin classes", frameon=False, loc="upper left", bbox_to_anchor=(0.00, 0.36), fontsize=11, title_fontsize=12)


def write_tables(rank_df: pd.DataFrame, enrichment_df: pd.DataFrame, recovery_df: pd.DataFrame, tables_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "supplementary_local_rank_rows": tables_dir / "supplementary_local_rank_rows.tsv",
        "supplementary_local_rank_enrichment_bins": tables_dir / "supplementary_local_rank_enrichment_bins.tsv",
        "supplementary_local_rank_recovery_curves": tables_dir / "supplementary_local_rank_recovery_curves.tsv",
    }
    rank_df.to_csv(paths["supplementary_local_rank_rows"], sep="\t", index=False)
    enrichment_df.to_csv(paths["supplementary_local_rank_enrichment_bins"], sep="\t", index=False)
    recovery_df.to_csv(paths["supplementary_local_rank_recovery_curves"], sep="\t", index=False)
    return paths


def plot_supplementary_figure(rank_df: pd.DataFrame, enrichment_df: pd.DataFrame, recovery_df: pd.DataFrame, figures_dir: pathlib.Path, *, max_predictions: int) -> pathlib.Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.8, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.95], wspace=0.30, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, enrichment_df)
    plot_panel_b(ax_b, rank_df)
    plot_panel_c(ax_c, recovery_df, max_predictions=max_predictions)
    plot_legend_panel(ax_legend)

    fig.suptitle("Functional target enrichment and recovery across local predictor ranks", fontsize=16, y=0.995)
    fig.canvas.draw()
    add_panel_label(fig, ax_a, "A")
    add_panel_label(fig, ax_b, "B")
    add_panel_label(fig, ax_c, "C")
    out_path = figures_dir / "supplementary_local_rank_enrichment_recovery.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplementary local-rank enrichment/recovery figure.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    parser.add_argument("--rank-bins", type=int, default=10)
    parser.add_argument("--max-predictions", type=int, default=500)
    args = parser.parse_args()

    rank_df = load_rank_table(args.report_dir, fdr_threshold=args.fdr_threshold, effect_threshold=args.effect_threshold)
    enrichment_df = enrichment_by_bin(rank_df, n_bins=args.rank_bins)
    recovery_df = recovery_curves(rank_df, max_predictions=args.max_predictions)
    outputs = write_tables(rank_df, enrichment_df, recovery_df, args.out_dir / "tables")
    outputs["supplementary_local_rank_figure"] = plot_supplementary_figure(
        rank_df,
        enrichment_df,
        recovery_df,
        args.out_dir / "figures",
        max_predictions=args.max_predictions,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
