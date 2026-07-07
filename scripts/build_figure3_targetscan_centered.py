"""Build reference-centered manuscript Figure 3.

This post-processing script creates the reference-rank Figure 3 used in the
manuscript. The figure now includes both TargetScan-centered and miRDB-centered
views in the same panel set.

For each reference predictor, pairs are first restricted to miRNA-gene pairs
scored by the reference predictor, binned by the reference local rank within each
perturbation dataset, and then each predictor is evaluated inside those same
reference-defined bins.

The reference local rank is recomputed from the reference score within each
joined dataset using average ranks for ties, normalized to 0-1 where 0 is the
weakest reference-scored pair and 1 is the strongest reference-scored pair.

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
REFERENCE_TOOLS = ["targetscan", "mirdb_mirtarget"]
TOOL_COLORS = {
    "targetscan": "#9467BD",
    "mirdb_mirtarget": "#FF7F0E",
    "microt_cnn": "#2CA02C",
    "mirbind2": "#D62728",
    "miraw": "#1F77B4",
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


def reference_centered_table(
    report_dir: pathlib.Path,
    *,
    reference_tool: str,
    fdr: float,
    effect_threshold: float,
    bins: int = 10,
) -> pd.DataFrame:
    """Calculate enrichment and coverage inside reference local-rank bins."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    reference_score_col = f"score_{reference_tool}"

    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        if reference_score_col not in frame.columns:
            continue

        reference_rank = local_rank_from_score(frame[reference_score_col])
        frame = frame.loc[reference_rank.notna()].copy()
        if frame.empty:
            continue
        frame["reference_local_rank"] = reference_rank.loc[frame.index].astype(float)

        for tool_id in TOOL_IDS:
            score_col = f"score_{tool_id}"
            scored = pd.to_numeric(frame[score_col], errors="coerce").notna() if score_col in frame.columns else pd.Series(False, index=frame.index)

            for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                in_bin_mask = (frame["reference_local_rank"] >= left) & (
                    (frame["reference_local_rank"] <= right) if i == bins - 1 else (frame["reference_local_rank"] < right)
                )
                in_bin = frame.loc[in_bin_mask]
                scored_in_bin = in_bin.loc[scored.loc[in_bin.index]]
                total = int(len(in_bin))
                scored_count = int(len(scored_in_bin))
                positives = int(scored_in_bin["is_positive"].sum())
                rows.append(
                    {
                        "dataset_id": path.parent.name,
                        "reference_tool_id": reference_tool,
                        "reference_predictor": TOOL_LABELS[reference_tool],
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
        raw.groupby(
            [
                "reference_tool_id",
                "reference_predictor",
                "tool_id",
                "predictor",
                "rank_bin_left",
                "rank_bin_right",
                "rank_bin_mid",
            ],
            as_index=False,
        )[["anchor_bin_total", "predictor_scored_count", "positive_count"]]
        .sum()
    )
    grouped["coverage_fraction"] = grouped["predictor_scored_count"] / grouped["anchor_bin_total"]
    grouped["positive_fraction_within_scored"] = grouped["positive_count"] / grouped["predictor_scored_count"]
    return grouped


def reference_line_style(tool_id: str, reference_tool: str) -> dict:
    """Use stable predictor colors; highlight the active reference by dash style."""
    if tool_id == reference_tool:
        return {
            "color": TOOL_COLORS[tool_id],
            "linestyle": "--",
            "linewidth": 2.6,
            "marker": "o",
            "markersize": 5.5,
            "zorder": 10,
        }
    return {
        "color": TOOL_COLORS[tool_id],
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 5.0,
        "zorder": 5,
    }


def plot_one_reference(ax_enrichment, ax_coverage, table: pd.DataFrame, *, reference_tool: str) -> None:
    reference_label = TOOL_LABELS[reference_tool]
    plot_order = [tool_id for tool_id in TOOL_IDS if tool_id != reference_tool] + [reference_tool]
    for tool_id in plot_order:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        kwargs = reference_line_style(tool_id, reference_tool)
        ax_enrichment.plot(
            sub["rank_bin_mid"],
            sub["positive_fraction_within_scored"] * 100,
            label=TOOL_LABELS[tool_id],
            **kwargs,
        )
        ax_coverage.plot(
            sub["rank_bin_mid"],
            sub["coverage_fraction"] * 100,
            label=TOOL_LABELS[tool_id],
            **kwargs,
        )

    ax_enrichment.set_title(f"{reference_label}-centered GT-positive enrichment")
    ax_enrichment.set_ylabel("GT positives among scored pairs (%)")
    ax_enrichment.set_ylim(bottom=0)
    style_axes(ax_enrichment)

    ax_coverage.set_title(f"Predictor coverage across {reference_label} bins")
    ax_coverage.set_ylabel("Pairs in bin scored by predictor (%)")
    ax_coverage.set_ylim(0, 105)
    style_axes(ax_coverage)


def plot_figure3(table: pd.DataFrame, figures_dir: pathlib.Path) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), sharex=True)

    for row, reference_tool in enumerate(REFERENCE_TOOLS):
        sub = table[table["reference_tool_id"] == reference_tool]
        plot_one_reference(axes[row, 0], axes[row, 1], sub, reference_tool=reference_tool)
        reference_label = TOOL_LABELS[reference_tool]
        axes[row, 0].set_xlabel(f"{reference_label} local rank bin (0 = weakest, 1 = strongest)")
        axes[row, 1].set_xlabel(f"{reference_label} local rank bin (0 = weakest, 1 = strongest)")

    axes[0, 1].legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    panel_labels = [["A", "B"], ["C", "D"]]
    for row in range(2):
        for col in range(2):
            axes[row, col].text(-0.12, 1.10, panel_labels[row][col], transform=axes[row, col].transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure3_reference_centered.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TargetScan- and miRDB-centered manuscript Figure 3.")
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

    tables = []
    for reference_tool in REFERENCE_TOOLS:
        tables.append(
            reference_centered_table(
                args.report_dir,
                reference_tool=reference_tool,
                fdr=args.fdr_threshold,
                effect_threshold=args.effect_threshold,
                bins=args.bins,
            )
        )
    table = pd.concat(tables, ignore_index=True)
    table_path = tables_dir / "figure3_reference_centered_summary.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    figure_path = plot_figure3(table, figures_dir)

    print(f"figure3: {figure_path}")
    print(f"figure3_table: {table_path}")


if __name__ == "__main__":
    main()
