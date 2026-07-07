"""Build the TargetScan-centered manuscript Figure 3.

This post-processing script creates the reference-rank Figure 3 used in the
manuscript. The analysis is intentionally TargetScan-centered: pairs are first
restricted to miRNA-gene pairs scored by TargetScan, binned by TargetScan local
rank within each perturbation dataset, and then each predictor is evaluated
inside those same TargetScan-defined bins.

The local TargetScan rank is recomputed from the TargetScan score within each
joined dataset using average ranks for ties, normalized to 0-1 where 0 is the
weakest TargetScan-scored pair and 1 is the strongest TargetScan-scored pair.

Example
-------
python scripts/build_figure3_targetscan_centered.py \
    --report-dir results/20260706_132519 \
    --out-dir manuscript_assets
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funmirbench.evaluate as ev

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
TOOL_COLORS = {
    "targetscan": "#000000",
    "mirdb_mirtarget": "#FF7F0E",
    "microt_cnn": "#2CA02C",
    "mirbind2": "#D62728",
    "miraw": "#9467BD",
}


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def style_axes(ax, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def save_figure(fig, out_path: pathlib.Path) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def usable_joined(frame: pd.DataFrame, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    work = ev._filter_usable_gt_rows(frame, fdr_threshold=fdr)
    work = ev._annotate_ground_truth(work)
    work["is_positive"] = ev._positive_mask(
        work,
        fdr_threshold=fdr,
        abs_logfc_threshold=effect_threshold,
    )
    return work


def local_rank_from_score(score: pd.Series) -> pd.Series:
    """Return local normalized rank in one dataset, using average ranks for ties."""
    values = pd.to_numeric(score, errors="coerce")
    ranks = values.rank(method="average", ascending=True)
    count = int(values.notna().sum())
    if count <= 0:
        return pd.Series(np.nan, index=score.index)
    if count == 1:
        return pd.Series(1.0, index=score.index, dtype=float)
    return (ranks - 1.0) / (float(count) - 1.0)


def targetscan_centered_table(
    report_dir: pathlib.Path,
    *,
    fdr: float,
    effect_threshold: float,
    bins: int = 10,
) -> pd.DataFrame:
    """Calculate enrichment and coverage inside TargetScan local-rank bins."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []

    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        if "score_targetscan" not in frame.columns:
            continue

        anchor_rank = local_rank_from_score(frame["score_targetscan"])
        frame = frame.loc[anchor_rank.notna()].copy()
        if frame.empty:
            continue
        frame["targetscan_local_rank"] = anchor_rank.loc[frame.index].astype(float)

        for tool_id in TOOL_IDS:
            score_col = f"score_{tool_id}"
            scored = pd.to_numeric(frame[score_col], errors="coerce").notna() if score_col in frame.columns else pd.Series(False, index=frame.index)

            for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                in_bin_mask = (frame["targetscan_local_rank"] >= left) & (
                    (frame["targetscan_local_rank"] <= right) if i == bins - 1 else (frame["targetscan_local_rank"] < right)
                )
                in_bin = frame.loc[in_bin_mask]
                scored_in_bin = in_bin.loc[scored.loc[in_bin.index]]
                total = int(len(in_bin))
                scored_count = int(len(scored_in_bin))
                positives = int(scored_in_bin["is_positive"].sum())
                rows.append(
                    {
                        "dataset_id": path.parent.name,
                        "tool_id": tool_id,
                        "predictor": TOOL_LABELS[tool_id],
                        "rank_bin_left": left,
                        "rank_bin_right": right,
                        "rank_bin_mid": (left + right) / 2,
                        "anchor_bin_total": total,
                        "predictor_scored_count": scored_count,
                        "positive_count": positives,
                        "coverage_fraction": scored_count / total if total else np.nan,
                        "positive_fraction_within_scored": positives / scored_count if scored_count else np.nan,
                    }
                )

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["tool_id", "predictor", "rank_bin_left", "rank_bin_right", "rank_bin_mid"], as_index=False)[
            ["anchor_bin_total", "predictor_scored_count", "positive_count"]
        ]
        .sum()
    )
    grouped["coverage_fraction"] = grouped["predictor_scored_count"] / grouped["anchor_bin_total"]
    grouped["positive_fraction_within_scored"] = grouped["positive_count"] / grouped["predictor_scored_count"]
    return grouped


def line_style(tool_id: str) -> dict:
    if tool_id == "targetscan":
        return {"color": "#000000", "linestyle": "--", "linewidth": 2.4, "marker": "o", "markersize": 5.5, "zorder": 10}
    return {"color": TOOL_COLORS[tool_id], "linestyle": "-", "linewidth": 2.0, "marker": "o", "markersize": 5.0, "zorder": 5}


def plot_figure3(table: pd.DataFrame, figures_dir: pathlib.Path) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharex=True)

    # Plot non-reference predictors first, then TargetScan last so the dashed reference line remains visible.
    plot_order = [tool_id for tool_id in TOOL_IDS if tool_id != "targetscan"] + ["targetscan"]
    for tool_id in plot_order:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        kwargs = line_style(tool_id)
        axes[0].plot(
            sub["rank_bin_mid"],
            sub["positive_fraction_within_scored"] * 100,
            label=TOOL_LABELS[tool_id],
            **kwargs,
        )
        axes[1].plot(
            sub["rank_bin_mid"],
            sub["coverage_fraction"] * 100,
            label=TOOL_LABELS[tool_id],
            **kwargs,
        )

    axes[0].set_title("TargetScan-centered GT-positive enrichment")
    axes[0].set_xlabel("TargetScan local rank bin (0 = weakest, 1 = strongest)")
    axes[0].set_ylabel("GT positives among scored pairs (%)")
    axes[0].set_ylim(bottom=0)
    style_axes(axes[0])

    axes[1].set_title("Predictor coverage across TargetScan bins")
    axes[1].set_xlabel("TargetScan local rank bin (0 = weakest, 1 = strongest)")
    axes[1].set_ylabel("Pairs in bin scored by predictor (%)")
    axes[1].set_ylim(0, 105)
    style_axes(axes[1])
    axes[1].legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure3_targetscan_centered.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create corrected TargetScan-centered manuscript Figure 3.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    figures_dir = args.out_dir / "figures"
    tables_dir = args.out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = targetscan_centered_table(
        args.report_dir,
        fdr=args.fdr_threshold,
        effect_threshold=args.effect_threshold,
        bins=args.bins,
    )
    table_path = tables_dir / "figure3_targetscan_centered_summary.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    figure_path = plot_figure3(table, figures_dir)

    print(f"figure3: {figure_path}")
    print(f"figure3_table: {table_path}")


if __name__ == "__main__":
    main()
