"""Build manuscript figures and tables from a FuNmiRBench report directory.

The branch `trial/manuscript-figures-tables` is intended to stay focused on
post-processing final benchmark reports into manuscript assets. This script does
not rerun the benchmark. It reads an existing report directory containing
`tables/*_per_experiment.tsv` files and per-dataset `joined.tsv` files, then
writes manuscript-ready PNG/SVG figures and TSV tables.

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
from matplotlib.ticker import PercentFormatter

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
METRICS = ["coverage", "positive_coverage", "aps", "auroc"]
METRIC_LABELS = {
    "coverage": "Coverage",
    "positive_coverage": "Positive coverage",
    "aps": "APS",
    "auroc": "AUROC",
}


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


def metric_rows(report_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        table = pd.read_csv(find_table(report_dir, f"{metric}_per_experiment.tsv"), sep="\t")
        id_cols = [col for col in table.columns if col not in TOOL_IDS]
        available_tools = [tool_id for tool_id in TOOL_IDS if tool_id in table.columns]
        long = table.melt(id_vars=id_cols, value_vars=available_tools, var_name="tool_id", value_name="value")
        long["metric"] = metric
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        rows.append(long.dropna(subset=["value"])[["tool_id", "metric", "value"]])
    return pd.concat(rows, ignore_index=True)


def write_summary_tables(metrics: pd.DataFrame, tables_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics.groupby(["tool_id", "metric"])["value"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    wide = summary.pivot(index="tool_id", columns="metric")
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()
    wide.insert(1, "predictor", wide["tool_id"].map(TOOL_LABELS).fillna(wide["tool_id"]))

    detailed = tables_dir / "table_s1_detailed_cross_dataset_predictor_summary.tsv"
    wide.to_csv(detailed, sep="\t", index=False)

    keep = ["tool_id", "predictor"]
    for metric in METRICS:
        for stat in ("mean", "median"):
            col = f"{metric}_{stat}"
            if col in wide.columns:
                keep.append(col)
    table1 = tables_dir / "table1_cross_dataset_predictor_summary.tsv"
    wide[keep].to_csv(table1, sep="\t", index=False)
    return {"table1": table1, "table_s1": detailed}


def plot_figure1(metrics: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, metric in zip(axes.ravel(), METRICS):
        sub = metrics[metrics["metric"] == metric]
        data = [sub.loc[sub["tool_id"] == tool_id, "value"].dropna().to_numpy(float) for tool_id in TOOL_IDS]
        labels = [TOOL_LABELS[tool_id] for tool_id in TOOL_IDS]
        ax.boxplot(data, patch_artist=True, showfliers=False)
        for idx, values in enumerate(data, start=1):
            if values.size:
                jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
                ax.scatter(np.full(values.size, idx) + jitter, values, s=18, alpha=0.65, edgecolors="white", linewidths=0.3)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        if metric in {"coverage", "positive_coverage"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        style_axes(ax)
    fig.suptitle("Cross-dataset predictor distributions", fontsize=15, y=1.0)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure1_cross_dataset_distributions.png")


def expected_effect(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    perturbation = frame.get("perturbation", pd.Series("", index=frame.index)).astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    oe = perturbation.isin(["OE", "OVEREXPRESSION"])
    kdko = perturbation.isin(["KO", "KD", "KNOCKOUT", "KNOCKDOWN"])
    effect.loc[oe] = -logfc.loc[oe]
    effect.loc[kdko] = logfc.loc[kdko]
    return effect


def dense_rank_scale(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(np.nan, index=values.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=values.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def rank_values(frame: pd.DataFrame, tool_id: str, rank_type: str) -> pd.Series | None:
    col = f"{rank_type}_rank_{tool_id}"
    score = f"score_{tool_id}"
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    if score in frame.columns:
        return dense_rank_scale(frame[score])
    return None


def usable_joined(frame: pd.DataFrame, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
    frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
    keep = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
    frame = frame.loc[keep].copy()
    frame["expected_effect"] = expected_effect(frame)
    frame["is_positive"] = (frame["FDR_num"] < fdr) & (frame["expected_effect"] > effect_threshold)
    return frame


def rank_fraction_table(report_dir: pathlib.Path, *, rank_type: str, bins: int, fdr: float, effect_threshold: float) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    all_tool_rows = []
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, rank_type)
            if ranks is None:
                continue
            tmp = pd.DataFrame({"tool_id": tool_id, "rank": ranks, "is_positive": frame["is_positive"]}).dropna(subset=["rank"])
            all_tool_rows.append(tmp)
    all_rows = pd.concat(all_tool_rows, ignore_index=True)
    for tool_id in TOOL_IDS:
        sub = all_rows[all_rows["tool_id"] == tool_id]
        overall = float(sub["is_positive"].mean()) if len(sub) else np.nan
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (sub["rank"] >= left) & ((sub["rank"] <= right) if i == bins - 1 else (sub["rank"] < right))
            in_bin = sub.loc[mask]
            total = int(len(in_bin))
            positives = int(in_bin["is_positive"].sum())
            rows.append({
                "rank_type": rank_type,
                "tool_id": tool_id,
                "predictor": TOOL_LABELS[tool_id],
                "rank_bin_start": left,
                "rank_bin_end": right,
                "rank_bin_mid": (left + right) / 2,
                "total_count": total,
                "positive_count": positives,
                "background_count": total - positives,
                "positive_fraction_within_bin": positives / total if total else np.nan,
                "overall_positive_fraction": overall,
            })
    return pd.DataFrame(rows)


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
        ax.plot(sub["rank_bin_mid"], sub["positive_fraction_within_bin"] * 100, marker="o", linewidth=2, label=TOOL_LABELS[tool_id])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GT positives within bin (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    style_axes(ax, grid_axis="both")


def plot_figure2(local: pd.DataFrame, global_: pd.DataFrame, recovery: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    plot_fraction(axes[0], local, title="Local rank enrichment", xlabel="Local normalized rank bin")
    plot_fraction(axes[1], global_, title="Global rank enrichment", xlabel="Global normalized rank bin")
    for tool_id in TOOL_IDS:
        sub = recovery[recovery["tool_id"] == tool_id]
        axes[2].plot(sub["prediction_count"], sub["recovery_fraction"] * 100, linewidth=2, label=TOOL_LABELS[tool_id])
    axes[2].set_title("GT-positive recovery")
    axes[2].set_xlabel("Predicted targets per dataset")
    axes[2].set_ylabel("Mean GT-positive recovery (%)")
    axes[2].set_ylim(bottom=0)
    style_axes(axes[2], grid_axis="both")
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Functional target enrichment and recovery across predictor ranks", fontsize=15, y=1.02)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure2_rank_enrichment_recovery.png")


def centered_table(report_dir: pathlib.Path, *, anchor_tool: str, fdr: float, effect_threshold: float, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
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
    for tool_id in TOOL_IDS:
        sub = data[data["tool_id"] == tool_id]
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (sub["anchor_rank"] >= left) & ((sub["anchor_rank"] <= right) if i == bins - 1 else (sub["anchor_rank"] < right))
            in_bin = sub.loc[mask]
            scored = in_bin[in_bin["scored"]]
            total = int(len(in_bin))
            scored_count = int(len(scored))
            positives = int(scored["is_positive"].sum())
            rows.append({
                "anchor_tool": anchor_tool,
                "anchor_label": TOOL_LABELS[anchor_tool],
                "tool_id": tool_id,
                "predictor": TOOL_LABELS[tool_id],
                "rank_bin_start": left,
                "rank_bin_end": right,
                "rank_bin_mid": (left + right) / 2,
                "anchor_bin_total": total,
                "predictor_scored_count": scored_count,
                "positive_count": positives,
                "background_count": scored_count - positives,
                "coverage_fraction": scored_count / total if total else np.nan,
                "positive_fraction_within_scored": positives / scored_count if scored_count else np.nan,
            })
    return pd.DataFrame(rows)


def plot_centered(table: pd.DataFrame, figures_dir: pathlib.Path, filename: str):
    anchor_label = table["anchor_label"].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 9.6), sharex=True)
    for tool_id in TOOL_IDS:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        axes[0].plot(sub["rank_bin_mid"], sub["positive_fraction_within_scored"] * 100, marker="o", linewidth=2, label=TOOL_LABELS[tool_id])
        axes[1].plot(sub["rank_bin_mid"], sub["coverage_fraction"] * 100, marker="o", linewidth=2, label=TOOL_LABELS[tool_id])
    axes[0].set_title(f"{anchor_label}-centered GT-positive enrichment")
    axes[0].set_ylabel("GT positives among scored genes (%)")
    axes[0].set_ylim(bottom=0)
    style_axes(axes[0], grid_axis="both")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[1].set_title(f"Predictor coverage across {anchor_label} bins")
    axes[1].set_xlabel(f"{anchor_label} local rank bin (0 = weakest, 1 = strongest)")
    axes[1].set_ylabel("Genes in bin scored by predictor (%)")
    axes[1].set_ylim(0, 105)
    style_axes(axes[1], grid_axis="both")
    fig.tight_layout()
    return save_figure(fig, figures_dir / filename)


def build_assets(report_dir: pathlib.Path, out_dir: pathlib.Path, *, fdr: float, effect_threshold: float):
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    metrics = metric_rows(report_dir)
    metrics.to_csv(tables_dir / "per_experiment_metric_rows.tsv", sep="\t", index=False)
    outputs.update(write_summary_tables(metrics, tables_dir))
    outputs["figure1"] = plot_figure1(metrics, figures_dir)

    local = rank_fraction_table(report_dir, rank_type="local", bins=10, fdr=fdr, effect_threshold=effect_threshold)
    global_ = rank_fraction_table(report_dir, rank_type="global", bins=5, fdr=fdr, effect_threshold=effect_threshold)
    recovery = recovery_table(report_dir, fdr=fdr, effect_threshold=effect_threshold)
    local.to_csv(tables_dir / "table_s4_local_rank_bins.tsv", sep="\t", index=False)
    global_.to_csv(tables_dir / "table_s5_global_rank_bins.tsv", sep="\t", index=False)
    recovery.to_csv(tables_dir / "table_s6_recovery_curves.tsv", sep="\t", index=False)
    outputs["figure2"] = plot_figure2(local, global_, recovery, figures_dir)

    targetscan = centered_table(report_dir, anchor_tool="targetscan", fdr=fdr, effect_threshold=effect_threshold)
    targetscan.to_csv(tables_dir / "table_s2_targetscan_centered_bins.tsv", sep="\t", index=False)
    outputs["figure3"] = plot_centered(targetscan, figures_dir, "figure3_targetscan_centered.png")

    mirdb = centered_table(report_dir, anchor_tool="mirdb_mirtarget", fdr=fdr, effect_threshold=effect_threshold)
    mirdb.to_csv(tables_dir / "table_s3_mirdb_centered_bins.tsv", sep="\t", index=False)
    outputs["supplementary_mirdb"] = plot_centered(mirdb, figures_dir, "supplementary_mirdb_centered.png")
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
