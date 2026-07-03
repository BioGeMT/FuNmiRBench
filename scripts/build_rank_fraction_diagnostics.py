"""Build complementary local/global rank-fraction diagnostics.

This utility creates presentation-friendly rank-bin plots that complement the
positive/background violin plots. Instead of plotting the positive and
background rank distributions independently, it bins normalized ranks and asks:

    GT positives in bin / all scored genes in bin

It also writes count-context plots/tables so sparse bins, especially in global
rank space, are visible.

Example
-------
python scripts/build_rank_fraction_diagnostics.py \
    --report-dir 20260629_135157 \
    --out-dir 20260629_135157/plots/ranks/refined_rank_fraction
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEFAULT_TOOL_IDS = ("targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw")
DEFAULT_TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}


def _expected_effect_from_logfc(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    if "perturbation" not in frame.columns:
        return logfc.abs()
    perturbation = frame["perturbation"].astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    effect.loc[perturbation.eq("OE")] = -logfc.loc[perturbation.eq("OE")]
    effect.loc[perturbation.isin(["KO", "KD"])] = logfc.loc[perturbation.isin(["KO", "KD"])]
    return effect


def _rank_scale_scores(scores: pd.Series) -> pd.Series:
    values = pd.to_numeric(scores, errors="coerce")
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(np.nan, index=scores.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=scores.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def _joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def _load_rank_rows(
    joined_paths: Iterable[pathlib.Path],
    *,
    rank_type: str,
    tool_ids: Iterable[str],
    tool_labels: dict[str, str],
    fdr_threshold: float,
    effect_threshold: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for joined_path in joined_paths:
        frame = pd.read_csv(joined_path, sep="\t")
        if not {"logFC", "FDR"}.issubset(frame.columns):
            continue
        frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
        frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
        usable = (
            frame["logFC_num"].notna()
            & frame["FDR_num"].notna()
            & (frame["FDR_num"] > 0.0)
            & (frame["FDR_num"] <= 1.0)
        )
        work = frame.loc[usable].copy()
        if work.empty:
            continue
        work["expected_effect"] = _expected_effect_from_logfc(work)
        work["is_positive"] = (
            (work["FDR_num"] < float(fdr_threshold))
            & (work["expected_effect"] > float(effect_threshold))
        )
        for tool_id in tool_ids:
            rank_col = f"{rank_type}_rank_{tool_id}"
            score_col = f"score_{tool_id}"
            if rank_col in work.columns:
                ranks = pd.to_numeric(work[rank_col], errors="coerce")
            elif score_col in work.columns:
                ranks = _rank_scale_scores(work[score_col])
            else:
                continue
            tmp = pd.DataFrame(
                {
                    "tool_id": tool_id,
                    "tool_label": tool_labels.get(tool_id, tool_id),
                    "rank_type": rank_type,
                    "rank_value": ranks,
                    "is_positive": work["is_positive"].astype(bool),
                }
            ).dropna(subset=["rank_value"])
            if not tmp.empty:
                rows.append(tmp)
    if not rows:
        return pd.DataFrame(
            columns=["tool_id", "tool_label", "rank_type", "rank_value", "is_positive"]
        )
    return pd.concat(rows, ignore_index=True)


def summarize_rank_fraction(rank_df: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    for (rank_type, tool_id, tool_label), sub in rank_df.groupby(
        ["rank_type", "tool_id", "tool_label"], sort=False
    ):
        total_overall = int(len(sub))
        positive_overall = int(sub["is_positive"].sum())
        overall_fraction = positive_overall / total_overall if total_overall else np.nan
        for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            if idx == len(bins) - 2:
                mask = (sub["rank_value"] >= left) & (sub["rank_value"] <= right)
            else:
                mask = (sub["rank_value"] >= left) & (sub["rank_value"] < right)
            in_bin = sub.loc[mask]
            total = int(len(in_bin))
            positives = int(in_bin["is_positive"].sum())
            background = total - positives
            rows.append(
                {
                    "rank_type": rank_type,
                    "tool_id": tool_id,
                    "tool_label": tool_label,
                    "n_bins": int(n_bins),
                    "rank_bin_start": float(left),
                    "rank_bin_end": float(right),
                    "rank_bin_mid": float((left + right) / 2.0),
                    "total_count": total,
                    "positive_count": positives,
                    "background_count": background,
                    "positive_fraction_within_bin": positives / total if total else np.nan,
                    "overall_positive_fraction": overall_fraction,
                }
            )
    return pd.DataFrame(rows)


def write_fraction_plot(summary_df: pd.DataFrame, out_path: pathlib.Path, *, title: str) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for tool_id, sub in summary_df.groupby("tool_id", sort=False):
        sub = sub.sort_values("rank_bin_mid")
        ax.plot(
            sub["rank_bin_mid"],
            sub["positive_fraction_within_bin"],
            marker="o",
            linewidth=2,
            label=str(sub["tool_label"].iloc[0]),
        )
        baseline = float(sub["overall_positive_fraction"].dropna().iloc[0])
        if np.isfinite(baseline):
            ax.hlines(baseline, xmin=0.0, xmax=1.0, linestyles="dotted", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Normalized rank bin (0 = weakest, 1 = strongest)")
    ax.set_ylabel("GT positives within bin")
    ax.set_title(title)
    ax.text(
        0.01,
        0.98,
        "Each point shows GT positives / all scored genes in the bin.\n"
        "Dotted lines show each predictor's overall GT-positive rate.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_count_context_plot(summary_df: pd.DataFrame, out_path: pathlib.Path, *, title: str) -> pathlib.Path:
    tool_order = summary_df[["tool_id", "tool_label"]].drop_duplicates()
    fig, axes = plt.subplots(
        len(tool_order),
        1,
        figsize=(9.5, max(2.0, 1.65 * len(tool_order) + 1.2)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    max_count = max(int(summary_df["total_count"].max()), 1)
    for ax, (_, tool_row) in zip(axes, tool_order.iterrows()):
        sub = summary_df[summary_df["tool_id"] == tool_row["tool_id"]].sort_values("rank_bin_mid")
        ax.bar(sub["rank_bin_mid"], sub["background_count"], width=0.16, label="Background")
        ax.bar(
            sub["rank_bin_mid"],
            sub["positive_count"],
            bottom=sub["background_count"],
            width=0.16,
            label="GT positive",
        )
        ax.set_ylabel(str(tool_row["tool_label"]), rotation=0, ha="right", va="center")
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.8)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Normalized rank bin (0 = weakest, 1 = strongest)")
    fig.suptitle(title, y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_rank_fraction_diagnostics(
    report_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    fdr_threshold: float = 0.05,
    effect_threshold: float = 1.0,
    local_bins: int = 10,
    global_bins: int = 5,
) -> dict[str, pathlib.Path]:
    paths = _joined_paths(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}

    for rank_type, n_bins in (("local", local_bins), ("global", global_bins)):
        rank_df = _load_rank_rows(
            paths,
            rank_type=rank_type,
            tool_ids=DEFAULT_TOOL_IDS,
            tool_labels=DEFAULT_TOOL_LABELS,
            fdr_threshold=fdr_threshold,
            effect_threshold=effect_threshold,
        )
        if rank_df.empty:
            continue
        summary_df = summarize_rank_fraction(rank_df, n_bins=n_bins)
        table_path = out_dir / f"{rank_type}_rank_positive_fraction_{n_bins}bins.tsv"
        summary_df.to_csv(table_path, sep="\t", index=False)
        written[f"{rank_type}_table"] = table_path
        written[f"{rank_type}_fraction_plot"] = write_fraction_plot(
            summary_df,
            out_dir / f"{rank_type}_rank_positive_fraction_{n_bins}bins.png",
            title=f"{rank_type.capitalize()} rank bins: fraction of genes that are GT positive ({n_bins} bins)",
        )
        written[f"{rank_type}_count_plot"] = write_count_context_plot(
            summary_df,
            out_dir / f"{rank_type}_rank_bin_counts_{n_bins}bins.png",
            title=f"{rank_type.capitalize()} rank-bin gene counts ({n_bins} bins)",
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    parser.add_argument("--local-bins", type=int, default=10)
    parser.add_argument("--global-bins", type=int, default=5)
    args = parser.parse_args()
    written = build_rank_fraction_diagnostics(
        args.report_dir,
        args.out_dir,
        fdr_threshold=args.fdr_threshold,
        effect_threshold=args.effect_threshold,
        local_bins=args.local_bins,
        global_bins=args.global_bins,
    )
    for key, path in written.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
